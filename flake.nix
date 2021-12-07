{
  description = "Calculating open system bath energy changes with HOPS and analytically.";

  inputs = {
    nixpkgs.url = "nixpkgs/nixos-unstable";
    mach-nix.url = "github:DavHau/mach-nix";
    flake-utils.url = "github:numtide/flake-utils";
    fcSpline.url = "github:vale981/fcSpline";
    stocproc.url = "github:vale981/stocproc";
  };

   outputs = { self, nixpkgs, flake-utils, mach-nix, fcSpline, stocproc }:
     let
       python = "python39";
       devShell = pkgs:
         pkgs.mkShell {
           buildInputs = [
             (pkgs.${python}.withPackages
               (ps: with ps; [ black mypy ]))
             pkgs.nodePackages.pyright
           ];
         };

     in flake-utils.lib.eachSystem ["x86_64-linux"] (system:
       let
         pkgs = nixpkgs.legacyPackages.${system};
         mach-nix-wrapper = import mach-nix { inherit pkgs python; };



         requirements = builtins.readFile ./requirements.txt;
         fcSplinePkg = fcSpline.defaultPackage.${system};
         stocprocPkg = stocproc.defaultPackage.${system};

         hopsflow  = mach-nix-wrapper.buildPythonPackage rec {
           src = ./.;
           requirements = ''
           numpy >= 1.20
           scipy >= 1.6
           h5py
           tqdm
           lmfit
           '';
           propagatedBuildInputs = [stocprocPkg fcSplinePkg];
         };

         pythonShell = mach-nix-wrapper.mkPythonShell {
           requirements = requirements;
           packagesExtra = [hopsflow];
         };

         mergeEnvs = envs:
           pkgs.mkShell (builtins.foldl' (a: v: {
             buildInputs = a.buildInputs ++ v.buildInputs;
             nativeBuildInputs = a.nativeBuildInputs ++ v.nativeBuildInputs;
           }) (pkgs.mkShell { }) envs);

       in {
         devShell = mergeEnvs [ (devShell pkgs) pythonShell ];
         defaultPackage = hopsflow;
         packages.hopsflow = hopsflow;
       });
}
