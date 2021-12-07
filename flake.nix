{
  description = "Calculating open system bath energy changes with HOPS and analytically.";

  inputs = {
    nixpkgs.url = "nixpkgs/nixos-unstable";
    mach-nix.url = "github:DavHau/mach-nix";
    flake-utils.url = "github:numtide/flake-utils";
    fcSpline.url = "github:vale981/fcSpline";
  };

   outputs = { self, nixpkgs, flake-utils, mach-nix, fcSpline }:
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

     in flake-utils.lib.eachSystem ["x86_64-linux"] (system:
       let
         pkgs = nixpkgs.legacyPackages.${system};
         mach-nix-wrapper = import mach-nix { inherit pkgs python pypiDataRev pypiDataSha256; };

         stocproc = (mach-nix-wrapper.buildPythonPackage
           {src = builtins.fetchTarball {
              url = "https://github.com/vale981/stocproc/archive/93589c45f7a4e1f43059139708d696ff5b066dd2.tar.gz";
              sha256 = "005q29g9yxng6d0w3dx19xn5mwbmf3nxmh3233d5d5wvar5xjzvr";
            };

            requirements = ''
                numpy
                scipy
                mpmath
                cython
            '';
            pname="stocproc";
            version = "1.0.1";
           });


         requirements = builtins.readFile ./requirements.txt;
         fcSplinePkg = fcSpline.defaultPackage.${system};

         hopsflow  = mach-nix-wrapper.buildPythonPackage {
           src=./.;
           propagatedBuildInputs = [fcSplinePkg];
           # packagesExtra = [hopsflow fcSplinePkg stocproc];
           # _.stocproc.buildInputs.add = [fcSplinePkg];
         };

         pythonShell = mach-nix-wrapper.mkPythonShell {
           requirements = requirements;
           packagesExtra = [hopsflow fcSplinePkg stocproc];
           _.stocproc.buildInputs.add = [fcSplinePkg];
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
