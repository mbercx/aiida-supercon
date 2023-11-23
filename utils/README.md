# Julia script to convert a Wannier90 chk file into EPW ukk file

To use this script, install
- julia
- inside this folder, run `julia --project=. -e 'using Pkg; Pkg.instantiate()'`.
    This will use the `Project.toml` and `Manifest.toml` files and make sure the
    exact same version of code is installed
- change the 1st line of `chk2ukk.jl`, to point to the exact path of this
    folder (on the supercomputer)
