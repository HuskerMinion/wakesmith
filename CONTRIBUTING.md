# Contributing to wakesmith

Fixes, improvements and test results are welcome: open an issue or a pull request.

- **License of contributions.** wakesmith is licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE).
  By sending a contribution you agree that it is licensed under those same terms, and that the author may
  also license it, along with the rest of wakesmith, under other terms later. Send only work you have the
  right to give on those terms.
- **Tests.** Run `uv run pytest` before sending code; add a test when you fix a bug.
- **Numbers.** If a change claims a better model, show held-out results (recordings the model never trained
  on), scored the way the device runs it (`wakesmith test`), next to the run before the change.
- **Data.** Don't attach recordings of other people without their consent, and don't add datasets whose
  license doesn't allow this use; `src/wakesmith/assets.py` lists every download with its license.
