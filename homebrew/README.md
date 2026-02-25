# ChipFlow Homebrew Tap

Homebrew formulae for ChipFlow tools.

## Installation

```bash
brew tap chipflow/tap
brew install vajax
```

## Contents

| Formula | Description |
|---------|-------------|
| `vajax` | GPU-accelerated analog circuit simulator with Verilog-A support |

## Notes

This directory contains the formula templates for the `ChipFlow/homebrew-tap` repository.
To set up the actual tap:

1. Create the `ChipFlow/homebrew-tap` repo on GitHub
2. Copy `Formula/` into it
3. Set up a `HOMEBREW_TAP_TOKEN` secret in the vajax repo (a PAT with repo scope on homebrew-tap)
4. The release workflow dispatches an event to update the formula SHA after PyPI publish
5. Add a `.github/workflows/update.yml` in homebrew-tap that handles the `repository_dispatch` event

### homebrew-tap update workflow

Create `.github/workflows/update.yml` in the `ChipFlow/homebrew-tap` repo:

```yaml
name: Update formula
on:
  repository_dispatch:
    types: [update-vajax]

jobs:
  update:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Update vajax formula
        env:
          VERSION: ${{ github.event.client_payload.version }}
        run: |
          URL="https://files.pythonhosted.org/packages/source/v/vajax/vajax-${VERSION}.tar.gz"
          SHA=$(curl -sL "$URL" | shasum -a 256 | cut -d' ' -f1)
          sed -i "s|url \".*\"|url \"${URL}\"|" Formula/vajax.rb
          sed -i "s|sha256 \".*\"|sha256 \"${SHA}\"|" Formula/vajax.rb
          sed -i "s|vajax-.*\.tar\.gz|vajax-${VERSION}.tar.gz|" Formula/vajax.rb

      - name: Commit and push
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add Formula/vajax.rb
          git commit -m "vajax ${VERSION}"
          git push
```
