{
  description = "Calculating open system bath energy changes with HOPS and analytically.";

  inputs = {
    nixpkgs.url = "nixpkgs/nixos-unstable";
    utils.url = "github:vale981/hiro-flake-utils";
  };

  outputs = { self, utils, nixpkgs, ... }:
    (utils.lib.poetry2nixWrapper nixpkgs {
      name = "hopsflow";
      shellPackages = pkgs: with pkgs; [ black pyright ];
      poetryArgs = {
        projectDir = ./.;
      };
    });
}
