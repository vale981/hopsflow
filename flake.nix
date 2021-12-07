{
  description = "Calculating open system bath energy changes with HOPS and analytically. ";

  inputs = {
    nixpkgs.url = "nixpkgs/nixos-unstable";
    mach-nix.url = "github:DavHau/mach-nix";
  };

  outputs = { self, nixpkgs, flake-utils, mach-nix }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        python = "python39";
        pkgs = nixpkgs.legacyPackages.${system};

        mach-nix-wrapper = import mach-nix { inherit pkgs python; };
        requirements = builtins.readFile ./requirements.txt;
        pythonBuild = mach-nix-wrapper.mkPython { inherit requirements; };
      in {
        devShell = pkgs.mkShell {
          buildInputs = [
            # dev packages
            (pkgs.${python}.withPackages
              (ps: with ps; [ black mypy sphinx-autodoc-typehints sphinx pydata-sphinx-theme  ])) # <--- change here
            pkgs.nodePackages.pyright

            # app packages
            pythonBuild
          ];
        };
      });
}
