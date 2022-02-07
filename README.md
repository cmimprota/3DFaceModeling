# smgp-3D-face-modeling-and-learning

## Organisation:
- Github: https://github.com/JonathanLehner/smgp-3D-face-modeling-and-learning.git

For more information on each part of the project, refer to the READMEs in the subfolders.

Also, refer to the slides for a more clear visualization of the results.

## Compiling

The parts of the projects use Cmake. To compile, in each subfolder you can use:

```
mkdir build; cd build
cmake -DCMAKE_BUILD_TYPE=Release ../
make
```

You will need to have libigl in the root directory or in the subdirectories to compile.


## Data outputs

Our outputs:

- [Cleaned input data with landmarks]
- [Rigidly aligned faces and scaled templates]
- [Warped templates]

[Cleaned input data with landmarks]: https://polybox.ethz.ch/index.php/s/hVw9myE4FGoWNTc
[Rigidly aligned faces and scaled templates]: https://polybox.ethz.ch/index.php/s/3pjv6hKjj9sBbXo
[Warped templates]: https://polybox.ethz.ch/index.php/s/I4vEA2s8Naws5jR


## Resources:
Data we used:
- [Provided inputs]
- [Folder shared with other teams] (pw: eigenfaces)

We annotated 12 of the 112 provided meshes and got the remaining anotations from the other groups.

[Provided inputs]: https://www.dropbox.com/sh/kvgxcbixbjsolt9/AABM1AHOr1AJnvz-qETJB0K0a?dl=0
[Folder shared with other teams]: https://polybox.ethz.ch/index.php/s/ZfYXXfV5SR4sQoB


Machine learning algorithms mentioned in slides:
- https://github.com/optas/latent_3d_points
- https://github.com/rusty1s/pytorch_geometric  
- https://coma.is.tue.mpg.de/ 

ML process:
- compare reconstruction errors
