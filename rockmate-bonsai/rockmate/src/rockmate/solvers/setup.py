from distutils.core import setup, Extension
setup(
    name="rockmate_csolver",
    version="1.0",
    ext_modules=[
        Extension(
            "rockmate_csolver",
            ["./rk_rotor/solver.c"],
            extra_compile_args=["-O3", "--std=c99"],
        ),
    ],
)

