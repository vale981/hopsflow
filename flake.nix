{
  description = "Calculating open system bath energy changes with HOPS and analytically.";

  inputs = {
    nixpkgs.url = "nixpkgs/nixos-unstable";
    utils.url = "github:vale981/hiro-flake-utils";
    # poetry2nix.url = "github:nix-community/poetry2nix";
    # utils.inputs.poetry.follows = "poetry2nix";
    utils.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = { self, utils, nixpkgs, ... }:
    (utils.lib.poetry2nixWrapper nixpkgs {
      name = "hopsflow";
      shellPackages = pkgs: with pkgs; [ black pyright ];
      python = pkgs: pkgs.python39;
      poetryArgs = {
        projectDir = ./.;
      };

      shellOverride = (pkgs: oldAttrs: {
        shellHook = ''
#                    export PYTHONPATH=/home/hiro/src/two_qubit_model/:$PYTHONPATH
                    export PYTHONPATH=/home/hiro/src/hops/:$PYTHONPATH
#                    export PYTHONPATH=/home/hiro/src/hopsflow/:$PYTHONPATH
#                    export PYTHONPATH=/home/hiro/src/stocproc/:$PYTHONPATH
                    '';
      });
    });
}
