{
  description = "Calculating open system bath energy changes with HOPS and analytically.";

  inputs = {
    nixpkgs.url = "nixpkgs/nixos-unstable";
    mach-nix.url = "github:DavHau/mach-nix";
    flake-utils.url = "github:numtide/flake-utils";
  };

   outputs = { self, nixpkgs, flake-utils, mach-nix }:
     let
       python = "python39";
       pypiDataRev = "master";
       pypiDataSha256 = "041rpjrwwa43hap167jy8blnxvpvbfil0ail4y4mar1q5f0q57xx";
       devShell = pkgs:
         pkgs.mkShell {
           buildInputs = [
             (pkgs.${python}.withPackages
               (ps: with ps; [ black mypy ]))
             pkgs.nodePackages.pyright
           ];
         };

     in flake-utils.lib.eachDefaultSystem (system:
       let
         pkgs = nixpkgs.legacyPackages.${system};
         mach-nix-wrapper = import mach-nix { inherit pkgs python pypiDataRev pypiDataSha256; };

         fcSpline = (mach-nix-wrapper.buildPythonPackage
           {src = builtins.fetchTarball {
              url = "https://github.com/cimatosa/fcSpline/archive/4312b2c63d52711bff1bfe282d92f98a3e5073fb.tar.gz";
              sha256 = "0l38q5avcbiyqlmmhznw9sg02y54fia6r7x2f9w6h3kqf2xh05yc";
            };
            pname="fcSpline";
            version="0.1";
            requirements=''
             numpy
             cython
             setuptools
             scipy
           '';
           });

         stocproc = (mach-nix-wrapper.buildPythonPackage
           {src = builtins.fetchTarball {
              url = "https://github.com/vale981/stocproc/archive/c81eead1b86d8da0caa5ec013b5fb65e9d3c3b79.tar.gz";
              sha256 = "00fvfmdcpkm9lp2zn8kzzn6msq7cypqhf87ihrf63ci5z4hg2jpl";
            };

            _.stocproc.builInputs.add = [fcSpline];
            pname="stocproc";
            version = "1.0.1";
            requirements = ''
             numpy
             cython
             setuptools
             mpmath
             scipy
             '';
           });

         requirements = builtins.readFile ./requirements.txt;


         pythonPackage = mach-nix-wrapper.buildPythonPackage {
           src=./.;
           _.hops.builInputs.add = [stocproc];
           packagesExtra = [stocproc fcSpline];
         };

         pythonShell = mach-nix-wrapper.mkPythonShell {
           requirements = requirements;
           packagesExtra = [pythonPackage];
         };

         mergeEnvs = envs:
           pkgs.mkShell (builtins.foldl' (a: v: {
             buildInputs = a.buildInputs ++ v.buildInputs;
             nativeBuildInputs = a.nativeBuildInputs ++ v.nativeBuildInputs;
           }) (pkgs.mkShell { }) envs);

       in {
         devShell = mergeEnvs [ (devShell pkgs) pythonShell ];
         defaultPackage = pythonPackage;
       });
}
