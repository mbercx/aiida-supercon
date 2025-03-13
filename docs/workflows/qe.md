# Quantum ESPRESSO

## `ElectronPhononWorkChain`

The `ElectronPhononWorkChain` is a linear workflow that calculates the Eliashberg spectral function based on the linear interpolation approach of Wierzbowska et al.[CITE].
It consists of the following five steps:

1. A first `pw.x` calculation, performed on a fine **k**-point that is used later to perform the linear interpolation.
1. A second `pw.x` calculation on a coarser grid, required to calculate the phonons in the next step.
1. The `ph.x` calculation which calculates the electron-phonon coefficients on the coarse **k**-grid with a commensurate **q**-grid.
1. The calculation of the real-space force constants using the `q2r.x` code.
1. Finally, use `matdyn.x` to interpolate over the **q**-grid and calculate the final electron-phonon coupling and corresponding spectral function.

```{figure} img/qe-electron-phonon.png
:alt: Schematic of the `ElectronPhononWorkChain`
:width: 300px
:align: center

Schematic of the `ElectronPhononWorkChain`.
```

Most of the default inputs of each calculation are gathered in the _protocol_ of the work chain [TODO: Think about how to present this here].
The main inputs that the user has to provide to the work chain are the following:

* `structure` [**required**]: input structure for which to calculate the electron-phonon coupling and Eliashberg spectral function.
  Note that the structure is used "as is", i.e. no relaxation or primitivization is performed by the work chain.
* `qpoints_distance` [_default_ = 0.5]: Density of the **q**-point mesh, defined in terms of the maximum permitted distance between two **q**-points in reciprocal space along each of the axes.
  The distance is expressed in terms of 1/Å.
* `kpoints_factor` [_default_ = 2]: 