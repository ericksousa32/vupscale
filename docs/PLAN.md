## Arquitetura Consolidada (ComfyUI VHS -> CLI + API Python Universal) — v2

A ferramenta (`vupscale`) traduz o fluxo em nós do ComfyUI (_VideoHelperSuite_) em um pipeline de _streaming_ agnóstico de ambiente, de uma **NVIDIA T4 (15 GB VRAM / 12.7 GB RAM / 2 vCPUs)** no Colab Free até **RTX 4090, L4, L40S, A100 ou H100** no RunPod e no Colab Pro.

**Princípio de design:** o comando mínimo precisa funcionar bem sem ajuste nenhum (`vupscale video.mp4 -m realesrgan-x4`). Todo o auto-tuning é sobrescrevível por flags, mas nenhuma flag é obrigatória além da entrada e do modelo.

|**Nó Original (ComfyUI)**|**Equivalente na vupscale**|**Parâmetros & Comportamento**|
|---|---|---|
|**Carregar Modelo de Upscale**|Loader `spandrel` + catálogo de aliases + precisão decidida pelo descritor do modelo|`--model` (alias, `.pth`, `.safetensors` ou URL), `--precision` (`auto`, `fp16`, `bf16`, `fp32`), `--compile` (`auto`, `on`, `off`)|
|**Meta Batch Manager**|Calibração empírica de memória + lotes de _tiles_ + OOM Recovery|`--batch-size` (`0` = auto), `--tile-size` (`0` = auto, `-1` = off), `--tile-overlap` (`32`)|
|**Load Video (Upload)**|Leitor `ffmpeg` em thread própria (`rawvideo rgb24`, `readinto` em buffer pré-alocado), seek rápido via `-ss`|`input` (posicional), `--fps`, `--max-frames`, `--skip-frames`, `--every-nth`, `--pre-width`/`--pre-height`|
|**Aumentar Resolução**|Inferência na GPU com upload `uint8`, conversão/escala de saída na GPU, `channels_last`, `cudnn.benchmark`|`--out-scale` / `--target-height` (redimensiona na GPU, ex.: modelo 4x → saída 2x)|
|**Video Combine**|Writer `ffmpeg` em thread própria, codec escolhido por teste real de encode, metadados de cor BT.709, checkpointing por segmentos com manifesto|`--output` (opcional), `--codec`, `--crf`/`--quality`, `--pix-fmt`, `--no-audio`, `--checkpoint-dir`, `--resume`|

### Fluxo de execução (3 estágios concorrentes)

```
[Thread Reader]  ffmpeg decode → readinto(buffer) → Queue(maxsize=N)
        ↓
[Main / GPU]     H2D uint8 (pinned, stream dedicado) → float/half na GPU → modelo
                 → resize opcional → clamp → uint8 na GPU → D2H (pinned)
        ↓
[Thread Writer]  Queue(maxsize=N) → stdin ffmpeg encode (+ troca de segmento)
```

Decode, inferência e encode rodam em paralelo. Na versão serial (ler → inferir → escrever), a GPU fica ociosa enquanto o FFmpeg codifica, e no Colab (2 vCPUs) isso pode custar 30–50% do throughput. O teto de RAM do host (15% da RAM livre) passa a limitar a **profundidade das filas** (`N`), que é o que de fato consome RAM.

---

## Revisão v2 — o que mudou e por quê

### Correções (bugs ou premissas incorretas no plano v1)

1. **A100 e H100 não têm NVENC.** Essas GPUs de datacenter só têm decodificadores (NVDEC). O codec `auto` precisa fazer um **encode de teste real** (`ffmpeg -f lavfi -i nullsrc=s=256x256 -frames:v 1 -c:v h264_nvenc -f null -`). Só olhar a lista de `ffmpeg -encoders` não basta, porque o encoder aparece listado mas falha. A T4 tem NVENC H.264/HEVC, mas **não AV1** (AV1 NVENC só existe a partir da Ada: L4, L40S, RTX 40xx).
2. **H.264 NVENC tem limite de 4096 px de largura.** 1080p × 4 = 7680×4320 falha com `h264_nvenc`. Com `auto`, use **HEVC quando a saída passar de 4096 px** (o HEVC NVENC suporta até 8192).
3. **Dessincronização de áudio.** `--skip-frames` sem aplicar o mesmo offset no áudio dessincroniza tudo. O áudio precisa de `-ss <skip/fps>` na entrada de áudio. Com `--every-nth > 1` (vídeo acelerado), o áudio deve ser **desativado com aviso**. O `--fps` só reamostra os frames sem mudar a duração, então o áudio continua sincronizado.
4. **Cores deslocadas (BT.601 vs BT.709).** A conversão `rgb24 → yuv420p` do FFmpeg usa BT.601 por padrão, o que deixa vídeos HD/4K com tons esverdeados ou avermelhados. No decoder: `scale=in_color_matrix=auto:in_range=auto`. No encoder: `scale=out_color_matrix=bt709:out_range=tv`, mais `-colorspace bt709 -color_primaries bt709 -color_trc bt709`.
5. **Precisão: fp16 em vez de bf16 como padrão nas Ampere+.** Para inferência de SR, o fp16 tem 10 bits de mantissa contra 7 do bf16, o que dá menos banding em gradientes, e a velocidade é igual nos Tensor Cores. O bf16 só ajuda quando o modelo estoura o alcance do fp16. A regra nova usa os campos do descritor do `spandrel`:
   - `supports_half` → `fp16` (qualquer GPU, inclusive T4);
   - senão, `supports_bfloat16` e `major >= 8` → `bf16`;
   - senão → `fp32`.
   - Alguns transformers (SwinIR/HAT/DAT) declaram `supports_half=False`. Forçar fp16 neles gera frames pretos ou NaN, e o plano v1 forçava fp16 na T4 sem checar isso.
6. **Padding manual é redundante.** O descritor do `spandrel` já aplica `size_requirements` (`multiple_of`, `minimum`, `square`) e remove o padding da saída quando chamado diretamente. Use isso, e respeite `descriptor.tiling` (`DISCOURAGED`/`INTERNAL` → tiling desligado no `auto`).
7. **`torch.compile(mode="reduce-overhead")` é frágil aqui.** Ele usa CUDA Graphs, que recompilam a cada mudança de shape (último lote menor, redução por OOM, tiles de borda). Troque por `mode="max-autotune-no-cudagraphs"` com shapes fixos (complete o último lote e os tiles de borda até o tamanho padrão). Ative no `auto` só quando o vídeo tiver frames suficientes para amortizar a compilação (ex.: > 1500 frames). Para um vídeo curto, compilar um HAT pode levar mais tempo que o render.
8. **`pin_memory()` a cada lote é caro.** Ele aloca memória pinada nova em cada chamada. Pré-aloque buffers pinados de entrada e saída uma vez e reutilize-os.
9. **Envie `uint8` para a GPU, não float.** Converter para float na CPU quadruplica o tráfego PCIe e o custo da CPU. Faça o inverso na saída: clamp e `uint8` **na GPU** antes do D2H. A saída é `scale²` vezes maior que a entrada, então esse é o maior gargalo de transferência.
10. **Ordem dos filtros compatível com o VHS.** No VHS, `force_rate` é aplicado antes de `skip_first_frames` e `select_every_nth`. A cadeia fica `fps → trim → select → scale`, com o `scale` por último para processar menos frames.
11. **Resume sem decodificar tudo de novo.** No v1, `trim=start_frame` decodifica e descarta todos os frames anteriores. Em vídeos CFR, use `-ss <frame/fps>` antes de `-i` (seek rápido e preciso com re-encode). Use `trim` só como fallback para VFR.
12. **Validação de segmento.** O `nb_frames` do ffprobe nem sempre está presente. Use `ffprobe -count_packets -select_streams v:0` e compare com a **contagem esperada** do segmento, não só com `> 0`.

### Otimizações de desempenho adicionadas

- **Pipeline de 3 estágios com filas limitadas** (ver diagrama acima). É o maior ganho isolado.
- **Leitura sem cópia:** `proc.stdout.readinto(memoryview(buffer))` direto num `np.ndarray` pré-alocado, com `bufsize` grande no `Popen`.
- **`torch.backends.cudnn.benchmark = True`** + `channels_last`: shapes fixos de vídeo são o caso ideal para isso.
- **Tiles em lote:** empilhe vários tiles numa única chamada ao modelo em vez de processá-los um a um. A utilização da GPU melhora muito na T4 com `tile_size=512`.
- **Calibração empírica de memória em vez de fórmula:** antes do render, rode 1 frame (ou 1 tile) com o `batch_size`/`tile_size` candidato, meça `torch.cuda.max_memory_allocated()` e extrapole o maior lote que cabe com 20–25% de margem. Uma fórmula de "bytes de ativação" erra muito entre arquiteturas (ESRGAN vs HAT varia em ordens de grandeza). Faça cache do resultado em `~/.cache/vupscale/calib.json`, por (GPU, modelo, resolução, dtype).
- **Redimensionamento na GPU** (`--out-scale`, `--target-height`): o caso comum é "modelo 4x, mas quero 1080p → 1440p". O `F.interpolate(..., mode="bicubic", antialias=True)` na GPU antes do D2H reduz a transferência e o encode.
- **Decode por hardware opcional** (`-hwaccel cuda` quando houver NVDEC): libera CPU no Colab, onde os 2 vCPUs são o gargalo real.
- **Gargalo de CPU no encode:** nas GPUs sem NVENC (A100/H100), o `libx264` em 4K/8K costuma ser o gargalo. O `auto` usa `-preset faster` para `libx264`/`libx265` quando houver poucos núcleos e mostra no dashboard qual estágio limita o throughput (GPU, decode ou encode), pela ocupação das filas.

### Melhorias de praticidade

- **Catálogo de modelos com alias:** `-m realesrgan-x4`, `realesrgan-x2`, `realesr-animevideov3`, `span-x2`... resolvidos para URL e cacheados. Caminhos locais e URLs continuam funcionando.
- **Saída opcional:** sem `-o`, gera `<input>_x4.mp4` ao lado da entrada (ou `_1440p` com `--target-height`).
- **`input` posicional** na CLI (`vupscale video.mp4 -m ...`), mantendo `-i` como alias.
- **Subcomandos utilitários:**
  - `vupscale doctor`: mostra ambiente, GPU, compute capability, NVENC funcional, versão do ffmpeg e configuração que seria escolhida (sem processar nada);
  - `vupscale models`: lista o catálogo e o cache;
  - `--dry-run`: imprime o plano completo (precisão, lote, tile, codec, filtros ffmpeg) e sai.
- **Entrada em lote:** aceita diretório ou glob (`vupscale "videos/*.mp4" -m ...`), carregando o modelo uma vez só.
- **Instalação simplificada:** como `torch` fica **sem pin de versão** no `pyproject.toml`, `pip install "git+https://github.com/ericksousa32/vupscale"` reaproveita o PyTorch do Colab e do RunPod sem baixar nada pesado (as dependências do `spandrel` já vêm nesses ambientes). O `setup_cloud.sh` fica só como fallback para instalar o `ffmpeg` e para quem clona o repositório.
- **Drive sempre em dois estágios:** mesmo sem `--checkpoint-dir`, se `--output` estiver sob `/content/drive`, grave em `/tmp` e copie no final (cópia para `.part` no destino + `os.replace`, que é atômico no mesmo filesystem).
- **Segmentos por tempo, não por frames fixos:** 500 frames de 4K com HAT podem levar 30 min, e uma queda do Colab perderia tudo isso. Use `--segment-seconds` (padrão `~180 s` de processamento) e nomeie os segmentos pelo intervalo (`seg_000000-000743.mp4`), para que o resume seja determinístico.
- **Manifesto de checkpoint** (`manifest.json`): hash da entrada (tamanho + mtime + primeiros MB), modelo e parâmetros que afetam a saída. O `--resume` recusa continuar se algo mudar, em vez de gerar um vídeo misturado.
- **Rich no Colab:** `refresh_per_second=2` e `transient=False`, para não inflar o output da célula.

### Novo: estratégia de testes (transversal às etapas)

- Vídeos sintéticos gerados na hora (`ffmpeg -f lavfi -i testsrc2=s=320x240:r=24 -f lavfi -i sine` → 2 s com áudio).
- Modelo minúsculo criado no teste (ex.: um `SPAN`/`Compact` aleatório salvo em `.pth`), para que a CI rode em **CPU** sem downloads.
- Testes-chave: contagem exata de frames em todas as combinações de filtros; sincronia do áudio com `skip_frames`; tiling sem costura (comparar tile vs. frame inteiro, `max abs diff` pequeno); resume após `kill` no meio de um segmento; OOM simulado (mock de `OutOfMemoryError`).
- GitHub Actions: `ruff` + `pytest` em CPU.

---

## Plano de Implementação em 5 Etapas (revisado)

### Etapa 1: Estrutura do Pacote, CLI/API Dupla, Catálogo de Modelos e Setup

- Layout `src/vupscale/` (Python ≥ 3.10), CLI `vupscale` (Typer) e API `from vupscale import upscale_video`.
- `torch`/`torchvision` sem pin de versão; instalação via `pip install git+...` funciona direto no Colab e no RunPod.
- `models.py`: catálogo de aliases → URL + download com cache.
- Subcomandos `upscale` (padrão), `doctor` e `models`.
- `scripts/setup_cloud.sh` só garante o `ffmpeg` e faz instalação editável quando o repo foi clonado.
- Base de testes (`tests/`, fixtures de vídeo sintético e modelo minúsculo) e CI.

### Etapa 2: Detecção de Hardware, Auto-Tuning por Calibração e OOM Recovery

- `RuntimeConfig` derivado do hardware **e** do descritor do modelo (`supports_half`, `supports_bfloat16`, `tiling`, `scale`).
- Precisão: `fp16` → `bf16` → `fp32` (ver correção 5).
- Tiling e lote por **calibração empírica** com cache, limitados pela VRAM (margem de 25%) e pelo teto de RAM das filas (15% da RAM livre).
- Codec por teste real de encode, com regra de resolução (H.264 ≤ 4096 px, senão HEVC).
- `AdaptiveBatchController`: em OOM, divide o lote pela metade até 1; depois ativa ou reduz o tile (`None → 768 → 512 → 256`); depois troca de precisão. Nenhum frame é perdido.

### Etapa 3: Engine de Inferência (`spandrel` + Tiling em Lote)

- Upload `uint8` pinado e reutilizado, conversão na GPU, `channels_last`, `cudnn.benchmark`, `inference_mode`.
- Tiles com sobreposição e máscara de rampa linear, processados **em lote** e completados até o tamanho fixo.
- Padding de tamanho delegado ao descritor (`size_requirements`).
- Resize de saída opcional na GPU, clamp e `uint8` na GPU, D2H pinado.
- `torch.compile` (`max-autotune-no-cudagraphs`) só quando compensa, com fallback eager.

### Etapa 4: Streaming de Vídeo e Áudio (FFmpeg Pipes, Threads)

- Reader e writer em threads com filas limitadas; `readinto` sem cópia; drenagem do `stderr` em `deque`.
- Filtros na ordem `fps → trim → select → scale`, com seek `-ss` para skip e resume (CFR).
- Áudio com o mesmo offset do vídeo; desativado com aviso quando `every_nth > 1`.
- Metadados de cor BT.709 corretos nas duas pontas.
- NVENC com `-rc vbr -cq <q> -preset p5 -tune hq`; CPU com `-crf`. `--quality` unificado (0–51), mapeado para `crf` ou `cq`.

### Etapa 5: Orquestrador, Checkpointing com Manifesto e Telemetria

- Pipeline de 3 estágios, dashboard Rich (FPS, VRAM, ETA e **estágio limitante**).
- Segmentos por tempo, nomeados por intervalo de frames, com `manifest.json`.
- Escrita em `/tmp` → `.part` no destino → `os.replace`; validação por `-count_packets` contra a contagem esperada.
- Concat final com `-f concat -c copy` + áudio remuxado com o offset correto.
- `try/finally` + tratamento de `SIGINT`: fecha o segmento atual de forma limpa (ele fica válido para o resume).

---

## Prompts Otimizados para Desenvolvimento com IA (v2)

Use os prompts em sequência. Eles incorporam todas as correções da revisão v2.

### Prompt 1 — Estrutura Base, CLI/API, Catálogo de Modelos e Setup

```
Atue como Engenheiro de Software Sênior especializado em Computer Vision, PyTorch e pipelines de vídeo de alta performance.

Projeto: `vupscale`, ferramenta Python (≥3.10) de upscaling de vídeo com modelos PyTorch (.pth/.safetensors) carregados via `spandrel`, inspirada no streaming do ComfyUI VideoHelperSuite, para Google Colab (incluindo T4 Free), RunPod e GPUs locais. Princípio: o comando mínimo `vupscale video.mp4 -m realesrgan-x4` deve funcionar bem sem nenhum ajuste.

Crie:
1. `pyproject.toml` (hatchling) com entrypoint `vupscale = "vupscale.cli:app"`. Dependências: `torch`, `torchvision` (SEM pin de versão, para reaproveitar o PyTorch pré-instalado do Colab/RunPod), `spandrel>=0.4`, `numpy`, `typer`, `rich`, `psutil`. Extras `dev`: `pytest`, `ruff`.
2. Módulos em `src/vupscale/`: `__init__.py` (exporta `upscale_video`), `cli.py`, `config.py` (dataclass `UpscaleOptions` com todos os parâmetros e defaults, fonte única de verdade para a CLI e a API), `models.py`, `hardware.py`, `engine.py`, `video_io.py`, `pipeline.py`, `console.py`.
3. `models.py`: catálogo `MODEL_ALIASES` (ex.: `realesrgan-x4` → RealESRGAN_x4plus.pth, `realesrgan-x2`, `realesr-animevideov3`, `realesr-general-x4v3`) com URL oficial; `resolve_model(spec) -> Path` aceita alias, caminho local ou URL; download com barra Rich para `~/.cache/vupscale/models/` (escrita em `.part` + `os.replace`, validação de `Content-Length`).
4. CLI (Typer):
   - Comando principal `upscale` (padrão quando o primeiro argumento é um arquivo): `input` posicional (arquivo, diretório ou glob; alias `-i`), `-o/--output` (opcional; padrão `<stem>_x{scale}.mp4` ou `_{H}p`), `-m/--model` (obrigatório).
   - Hardware: `--device` (auto|cuda|cpu), `--precision` (auto|fp16|bf16|fp32), `--batch-size` (0=auto), `--tile-size` (0=auto, -1=off), `--tile-overlap` (32), `--compile` (auto|on|off).
   - Saída: `--out-scale` (float, 0=nativo do modelo), `--target-height` (0=off).
   - Frames (estilo VHS): `--fps` (0=original), `--max-frames` (0), `--skip-frames` (0), `--every-nth` (1), `--pre-width` (0), `--pre-height` (0).
   - Encoding: `--codec` (auto|h264|hevc|av1|h264_nvenc|hevc_nvenc|av1_nvenc), `--quality` (20; vira crf na CPU e cq no NVENC), `--pix-fmt` (yuv420p), `--no-audio`.
   - Resiliência: `--checkpoint-dir`, `--resume`, `--segment-seconds` (180).
   - `--dry-run`: imprime a configuração resolvida e sai.
   - Subcomandos `doctor` (ambiente, GPU, compute capability, NVENC funcional, versão do ffmpeg) e `models` (lista o catálogo e o cache).
5. `upscale_video(input, model, output=None, **opções)` com a mesma semântica da CLI, usando `UpscaleOptions`.
6. `scripts/setup_cloud.sh` (idempotente): instala `ffmpeg` via apt se ausente; se executado dentro do repo clonado, `pip install -e .` (pip mantém o torch existente porque não há pin). Documente no README que, no Colab, basta `pip install git+https://github.com/ericksousa32/vupscale`.
7. `tests/conftest.py` com fixtures: vídeo sintético de 2 s (`testsrc2` + `sine`) e um modelo SR minúsculo de pesos aleatórios salvo em `.pth` (arquitetura reconhecida pelo spandrel, ex.: Compact/SRVGGNet x2), para a CI rodar em CPU. Workflow GitHub Actions com ruff + pytest.

Os módulos ainda não implementados podem ter stubs com `NotImplementedError`, mas a CLI precisa rodar `--help`, `doctor` e `models`.
```

### Prompt 2 — Hardware, Auto-Tuning por Calibração e OOM Recovery (`hardware.py`)

```
Implemente `src/vupscale/hardware.py`.

1. Dataclass `RuntimeConfig`: environment ("colab"|"runpod"|"local"), device, dtype, batch_size, tile_size (int|None), tile_overlap, compile_model, video_codec, gpu_name, compute_cap, vram_total_gb, vram_free_gb, ram_free_gb, cpu_count, has_nvenc (dict codec→bool), queue_depth.
2. Detecção de ambiente: colab (`COLAB_RELEASE_TAG` em os.environ ou /content existe), runpod (`RUNPOD_POD_ID` ou /workspace), senão local.
3. Precisão (auto), usando o descritor do spandrel:
   - CPU → float32.
   - CUDA: se `descriptor.supports_half` → float16; senão, se `descriptor.supports_bfloat16` e major>=8 → bfloat16; senão → float32.
   - Override manual respeitado, mas com aviso se o modelo declarar que não suporta o dtype pedido.
4. Codec (auto): teste REAL de encode para cada NVENC (`ffmpeg -hide_banner -f lavfi -i nullsrc=s=256x256 -frames:v 1 -c:v <enc> -f null -`, timeout 10 s, com cache na sessão). Lembre que A100/H100 não têm NVENC e que a T4 não tem av1_nvenc. Regra: se a largura ou a altura de saída > 4096 → hevc_nvenc (ou libx265); senão h264_nvenc (ou libx264). Com CPU e cpu_count <= 4, use preset "faster".
5. `torch.compile` (auto): só em CUDA com major>=8, fora do Colab Free e quando o total estimado de frames > 1500.
6. Calibração empírica `calibrate(engine, frame_hw, cfg)`:
   - Se `descriptor.tiling` for DISCOURAGED/INTERNAL, tiling desligado no modo auto.
   - Tente processar 1 frame inteiro (dummy uint8) com batch=1; meça `torch.cuda.max_memory_allocated()` (após `reset_peak_memory_stats`). Se der OOM ou passar de 75% da VRAM livre, use tile 512 (múltiplo de 64) e meça de novo; continue reduzindo se necessário.
   - Extrapole o batch_size máximo (pico por item, margem de 25%), limitado a 16.
   - queue_depth: maior N tal que 2·N·batch·(bytes_entrada + bytes_saída) ≤ 15% da RAM livre (mínimo 1).
   - Cache em `~/.cache/vupscale/calib.json`, por (gpu_name, nome/hash do modelo, HxW, dtype, versão do torch).
7. `AdaptiveBatchController`: `run(fn, batch)` captura `torch.cuda.OutOfMemoryError`, executa `gc.collect()` + `torch.cuda.empty_cache()` e degrada nesta ordem: batch/2 até 1 → tile None→768→512→256 → (se fp32 por escolha, nada; se bf16/fp16 não for possível, falha com mensagem clara). Divide o lote corrente em sub-lotes e devolve os resultados na ordem original. Nenhum frame é perdido. Registra cada degradação para exibir no dashboard.

Código tipado, sem dependência de GPU para importar (a CI roda em CPU).
```

### Prompt 3 — Engine de Inferência (`engine.py`)

```
Implemente `src/vupscale/engine.py`.

1. Carregue com `spandrel.ModelLoader().load_from_file(path)`. Exija `ImageModelDescriptor` com purpose in ("SR", "Restoration"); caso contrário, erro claro. Exponha `scale`, `supports_half`, `supports_bfloat16`, `tiling`, `size_requirements` e o nome da arquitetura.
2. `UpscaleEngine(descriptor, cfg)`:
   - `.eval()`, `.to(device, dtype)`, `memory_format=torch.channels_last`; `torch.backends.cudnn.benchmark = True`.
   - Se `cfg.compile_model`: `torch.compile(descriptor.model, mode="max-autotune-no-cudagraphs", dynamic=False)` em try/except, com fallback eager e aviso.
   - Use a chamada do descritor (que aplica `size_requirements`: padding e recorte automáticos). Não reimplemente padding.
   - Pré-aloque buffers pinados de entrada (uint8) e saída (uint8) para o batch máximo e reutilize-os. Use um `torch.cuda.Stream` dedicado para H2D/D2H.
3. `process_batch(frames: np.ndarray) -> np.ndarray` ((B,H,W,3) uint8 RGB):
   - Copie para o buffer pinado, envie como uint8 (`non_blocking=True`) e, na GPU: permute → dtype → /255 → channels_last.
   - `torch.inference_mode()`.
   - Sem tile: modelo no batch inteiro. Para shapes fixos (cudnn/compile), complete o último lote até o batch padrão e descarte o excedente.
   - Com tile: gere todas as janelas (tile_size, overlap) de todos os frames do lote; complete os tiles de borda até tile_size com padding reflect; processe os tiles EM LOTES (sub-batch calibrado); acumule com máscara de rampa linear 2D (pesos 0→1 na faixa de overlap, escalada pelo fator do modelo) em um acumulador float32 na GPU, junto com o acumulador de pesos; normalize.
   - Resize opcional (out_scale/target_height) na GPU: `F.interpolate(mode="bicubic", antialias=True)`.
   - Clamp [0,1] → *255 → round → uint8 NA GPU → permute para (B,H,W,3) → D2H no buffer pinado → retorne uma view/cópia numpy.
4. Teste: a saída com tile (tile pequeno, overlap 8) vs. sem tile no modelo minúsculo deve ter `max abs diff` ≤ 2 (uint8), sem padrão de grade.
```

### Prompt 4 — Streaming de Vídeo e Áudio (`video_io.py`)

```
Implemente `src/vupscale/video_io.py` com subprocess e pipes (nada de frames em disco).

1. `probe_video(path) -> VideoMetadata`: width, height, fps_num, fps_den, fps_float, is_vfr (r_frame_rate ≠ avg_frame_rate), total_frames (nb_frames, ou `-count_packets` como fallback), duration, has_audio, audio_codec, color_space/primaries/transfer/range (para propagar), pix_fmt.
2. `build_decode_filters(opts, meta) -> str`, na ordem do VHS: `fps={fps}` → `trim=start_frame={skip}` (só quando o seek -ss não puder ser usado) → `select='not(mod(n\,{nth}))'` → `scale={w}:{h}:flags=lanczos` (-2 no eixo 0). No final, `scale=in_color_matrix=auto:in_range=auto,format=rgb24`. Calcule e retorne também o número esperado de frames e o fps efetivo de saída.
3. `VideoStreamReader` (context manager + thread):
   - Seek rápido: se skip_frames > 0 e o vídeo for CFR, use `-ss {skip/fps}` antes de `-i`; senão, trim.
   - `-hwaccel cuda` opcional (só para decode, com fallback se falhar).
   - `-frames:v {max_frames}` quando > 0.
   - Thread produtora: `stdout.readinto(memoryview(buf))` em arrays pré-alocados (pool reciclado), montando lotes (N,H,W,3) e colocando em `queue.Queue(maxsize=queue_depth)`. Sentinela no EOF; exceções propagadas para a thread principal.
   - Iterador `iter_batches()` que consome a fila.
4. `VideoStreamWriter` (context manager + thread consumidora com fila limitada):
   - stdin rawvideo rgb24 (WxH, fps de saída); conversão `scale=out_color_matrix=bt709:out_range=tv,format={pix_fmt}` + `-colorspace bt709 -color_primaries bt709 -color_trc bt709`.
   - Codecs: h264→libx264, hevc→libx265 (+ `-tag:v hvc1`), av1→libsvtav1 (se ausente, libaom-av1). CPU: `-crf {quality} -preset {preset}`. NVENC: `-rc vbr -cq {quality} -b:v 0 -preset p5 -tune hq`.
   - Áudio: quando habilitado, `-ss {audio_offset}` na segunda entrada; `-map 0:v:0 -map 1:a:0?`, `-c:a copy` com fallback automático (re-executar o mux) para `aac -b:a 192k`; `-shortest`. Desative o áudio com aviso se every_nth>1 (o filtro fps preserva a duração, então o áudio segue em sincronia).
   - `-movflags +faststart`. Garanta largura/altura pares (pad de 1 px se necessário).
   - Drene o stderr em thread daemon para `deque(maxlen=200)`; em erro, levante uma exceção com as últimas linhas.
5. `concat_segments(segments, output, audio_source, audio_offset, no_audio)`: lista para `-f concat -safe 0`, `-c:v copy`, áudio com o offset correto; escrita em `.part` + `os.replace`.
6. `count_frames(path)` via `ffprobe -count_packets -select_streams v:0`.
7. Testes: contagem exata de frames para combinações de skip/nth/max/fps; áudio presente e duração coerente; cores (um frame de cor sólida conhecida passa pelo ciclo decode→encode→decode com erro ≤ 3).
```

### Prompt 5 — Orquestrador, Checkpointing e Dashboard (`pipeline.py`)

```
Implemente `src/vupscale/pipeline.py` integrando hardware, engine e video_io, e conecte à CLI e à API.

1. `console.py`: Console Rich compartilhado; em Jupyter/Colab, `force_jupyter=True` e Progress com `refresh_per_second=2`.
2. Painel inicial: ambiente, GPU + compute capability, precisão, batch, tile, modelo (arquitetura + scale), resolução de entrada → saída, codec, filtros, frames esperados, áudio (on/off e motivo).
3. Progress: barra, frames atual/total, FPS, VRAM usada/total, ETA e "gargalo" (decode/GPU/encode, inferido pela ocupação das filas: fila de entrada vazia → decode; fila de saída cheia → encode).
4. Loop: reader → `AdaptiveBatchController.run(engine.process_batch, lote)` → writer, com as três etapas concorrentes.
5. Checkpointing (quando houver checkpoint_dir, ou automaticamente quando a saída estiver em /content/drive):
   - `manifest.json`: hash da entrada (tamanho, mtime, sha1 dos primeiros 8 MB), modelo (sha1), parâmetros que afetam a saída e versão do vupscale.
   - Segmentos por tempo de processamento (`segment_seconds`), cortados na fronteira de lote e nomeados `seg_{inicio:06d}-{fim:06d}.mp4`.
   - Cada segmento é escrito em `/tmp/vupscale_work/`; ao fechar: cópia para `<checkpoint_dir>/<nome>.part` + `os.replace`.
   - `--resume`: valida o manifesto (aborta com mensagem clara se divergir); valida cada segmento contando os frames com `count_frames` contra o intervalo do nome; remove os inválidos; retoma do primeiro frame faltante via seek; no final, `concat_segments` → `--output`.
   - Sem checkpoint e com saída no Drive: escreve em /tmp e copia no final.
6. Encerramento: `try/finally` fecha os pipes, encerra as threads e mata o ffmpeg se necessário. No `KeyboardInterrupt`, finaliza o segmento atual de forma limpa para que ele seja reaproveitável no resume.
7. Ao final: resumo com tempo total, FPS médio, degradações por OOM e o caminho da saída.
8. Testes de integração (CPU, modelo minúsculo): render completo; resume após interrupção simulada no meio (resultado com contagem de frames exata e sem frames duplicados); manifesto divergente → erro.
```

---

## Exemplos Práticos de Execução

**1. RunPod (terminal em `/workspace`):**

```bash
pip install "git+https://github.com/ericksousa32/vupscale"
vupscale doctor
vupscale /workspace/input.mp4 -m realesrgan-x2
# → /workspace/input_x2.mp4
```

**2. Google Colab (célula Python com Google Drive):**

```python
from google.colab import drive
drive.mount('/content/drive')

!pip install -q "git+https://github.com/ericksousa32/vupscale"

from vupscale import upscale_video

upscale_video(
    "/content/drive/MyDrive/videos/paramore.mp4",
    model="realesrgan-x4",
    target_height=2160,            # modelo 4x, saída final 4K
    checkpoint_dir="/content/drive/MyDrive/vupscale_ckpt",
    resume=True,
)
```

**3. Pré-visualizar a configuração sem processar:**

```bash
vupscale clip.mp4 -m realesrgan-x4 --dry-run
```
