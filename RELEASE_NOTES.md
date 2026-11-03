## What's new in 0.4.7

**A smith at the forge while you train.** The Training page shows a Norse smith working on your wake
word while a run goes: he stirs the crucible while the synthetic voices are made, sorts a heap of
letter ingots while your recordings are prepared, heats the bar at the forge (a raven works the
bellows), hammers it on the anvil while the models train, inspects the blade during the test and
quenches it at export. A small dial shows how far the current step is. When the run is done he holds
up the finished blade with your phrase on it; if it stops, he lays the hammer down. With reduced
motion turned on in your system, he holds still.

**`wakesmith test --model` keeps the project's results.** Scoring another model file, such as the one
on your devices, used to replace the project's own results with that file's scores. They are now saved
under `results/files/`, named by the file's checksum, and the command says where. Say which kind of
model the file is with `--kind mww` or `--kind oww`.

**`wakesmith setup` finds uv where its installer puts it.** Started from a service or a cron job whose
PATH lacks `~/.local/bin`, setup fell back to pip, which an environment made by uv doesn't have, and
stopped with a long error. It now looks in `~/.local/bin` too, and if there's neither uv nor pip it says
so in one line before removing anything.

**openWakeWord feature workers share the cores** too: each runs one OpenBLAS thread, like the
microWakeWord workers since 0.4.6.

To update: `uv tool install --python 3.12 --force ./wakesmith-0.4.7-py3-none-any.whl`.

Earlier: Macs with Apple silicon and the folder summary in
[0.4.6](https://github.com/HuskerMinion/wakesmith/releases/tag/v0.4.6), Also written as, play buttons that show what's playing and the version's folder in
[0.4.5](https://github.com/HuskerMinion/wakesmith/releases/tag/v0.4.5), Wake words page menu fixes in
[0.4.4](https://github.com/HuskerMinion/wakesmith/releases/tag/v0.4.4), macOS training fixes (workers, RAM, open-file limit) in
[0.4.3](https://github.com/HuskerMinion/wakesmith/releases/tag/v0.4.3), macOS job start and Training page fixes in
[0.4.2](https://github.com/HuskerMinion/wakesmith/releases/tag/v0.4.2), install fixes in
[0.4.1](https://github.com/HuskerMinion/wakesmith/releases/tag/v0.4.1), German wake words (experimental) in
[0.4.0](https://github.com/HuskerMinion/wakesmith/releases/tag/v0.4.0).
