# h13 — overnight resume

The **phone** queue runs continuously and needs nothing from you; it only uses the
Mac for `devicectl` polling. The **Mac control arm** is the job that competes for
your machine, so it is paused during the day and resumed overnight.

## Resume the Mac control arm (run this before bed)

```
cd ~/Documents/Research/mobile_LoRA_ft
nohup caffeinate -is .venv-mlx/bin/python train/train_user_mlx_h13.py \
  --all --skip-existing > /tmp/h13_mac_train.log 2>&1 &
```

`--skip-existing` keys on each user's saved `adapters.safetensors`, so it resumes
where it stopped and a half-finished user is simply retrained. Unthrottled it needs
about 6 h for all 100 users. Add `--throttle 6 --cache-limit-gb 2 --nice 5` if you
want it polite again (about 15 h, still ~3x ahead of the phone).

Progress: `ls train/checkpoints_mlx/h13_mac_control | wc -l`

## Next morning: fill in any missing `mac` arms

The phone evaluates whatever adapters exist, so users it reached before the Mac did
have `rag/cluster/device` but no `mac`. This adds only that arm:

```
nohup scripts/h13/run_h13_mac_catchup.sh > /tmp/h13_catchup.log 2>&1 &
```

One extra model load (~1 min) per user. Far cheaper than stalling the phone.

## One deferred check (needs the Mac GPU, ~5 min)

The on-device RAG arm for user `8000201` scored 0.285 where the bf16 cluster arm
scored 0.5155. Confirm that is the quantised model's honest score rather than a
device-side artefact, by scoring the same 452 prompts with the Mac 4-bit model:

```
.venv-mlx/bin/python scripts/h13/diag_eval_plane.py --n-users 1 \
  --out results/ondevice/h13_diag_user8000201.json
```

(Edit the script's user slice if rank 0 is not `data[0]`.) The two planes agreed
exactly on the 15-user diagnostic, so this is expected to confirm.

## Check the campaign at any time

```
.venv-mlx/bin/python eval/h13_score.py --require-arms rag,cluster,device
tail -5 /tmp/h13_queue.log
```

Any prefix of the frozen queue is a complete, reportable result. **Always report the
prefix bias table from the results doc alongside a prefix number** — the queue
front-loads high-effect users (cluster Δ +0.283 at prefix 1 vs +0.076 at 100).

## If the phone queue dies

It is restartable and skips completed users:

```
nohup caffeinate -ims scripts/h13/run_h13_queue.sh > /tmp/h13_queue.log 2>&1 &
```
