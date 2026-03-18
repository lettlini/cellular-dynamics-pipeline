{
  description = "Dev shell with conda for nextflow analysis";

  outputs = {
    self,
    nixpkgs,
    ...
  }: let
    system = "x86_64-linux";
    pkgs = nixpkgs.legacyPackages.${system};
  in {
    devShells.${system}.default = let
      libPath = pkgs.lib.makeLibraryPath [
        # pkgs.wayland
        # pkgs.vulkan-loader
      ];
    in
      (pkgs.buildFHSEnv {
        name = "cell-tracking-nextflow-analysis-shell";

        targetPkgs = _: [
          pkgs.micromamba
          pkgs.nextflow
          pkgs.libxcb
          pkgs.libxcb-util
          pkgs.libGL
          pkgs.libxkbcommon
          pkgs.glib
        ];

        profile = ''
          # Create or activate env inside the dev shell
          if [ ! -d ./.conda-env ]; then
            micromamba create -y -p ./.conda-env
          fi
          eval "$(micromamba shell hook bash)"
          micromamba activate ./.conda-env
          micromamba install -y conda
        '';
      }).env;
  };
}
