{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
  }:
    flake-utils.lib.eachDefaultSystem (system: let
      pkgs = nixpkgs.legacyPackages.${system};
      manifest = (pkgs.lib.importTOML ./pyproject.toml).tool.poetry;

      buildPkgs = with pkgs; [
        pkg-config
        cairo
        ninja
        gobject-introspection
      ];

      pyBuildPkgs = with pkgs.python3.pkgs; [
        setuptools
        poetry-core
      ];

      libPkgs = with pkgs.python3.pkgs; [
        numpy
        matplotlib
        pygobject3
        opencv4
      ];

      devPkgs = with pkgs; [
        poetry
        just
        watchexec
        gcc
        stdenv.cc.cc
      ];

      setupPkgConfigPath = pkgs.lib.makeSearchPathOutput "lib" "pkgconfig" buildPkgs;

    in {
      defaultPackage = pkgs.python3.pkgs.buildPythonApplication {
        pname = manifest.name;
        version = manifest.version;
        format = "pyproject";

        src = pkgs.lib.cleanSource ./.;

        nativeBuildInputs = buildPkgs ++ pyBuildPkgs;
        propagatedBuildInputs = libPkgs;

        preCheck = ''
          export HOME=$(mktemp -d)
        '';
      };

      devShell = pkgs.mkShell {
        nativeBuildInputs = buildPkgs ++ pyBuildPkgs;
        propagatedBuildInputs = libPkgs;
        buildInputs = devPkgs;

        shellHook = ''
          poetry install > /dev/null
          VENV_PATH=$(poetry env info --path)
          export PATH="$VENV_PATH/bin:$PATH"
          export LD_LIBRARY_PATH=$(echo ${pkgs.lib.makeLibraryPath devPkgs}):$LD_LIBRARY_PATH
          export PKG_CONFIG_PATH=${setupPkgConfigPath}:$PKG_CONFIG_PATH
        '';
      };
    });
}
