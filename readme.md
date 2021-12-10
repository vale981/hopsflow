This repo contains code used in the work for my master thesis.

# Docs
Documentation is available [here](https://vale981.github.io/hopsflow/).

# Installation
## Nix
For developing use `nix develop` and for installing use the default
package from the flake.

If you plan to use this package in another poetry2nix project you have
to include the overrides from `lib.overrides` in the flake
`github:vale981/hiro-flake-utils`.

## Poetry
For development use `poetry shell` and for installing just add this
repo to the depencies of your project. You can also build a standalone
package with `poetry build` using.
