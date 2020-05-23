# vai

- Make sure to create Vitis AI-enabled platform

## Preparation

### Generate .dcf file

```
$ dlet -f <input .hwh file>
$ mkdir arch/<board name>
$ mv <generated file> arch/<board name>/<board name>.dcf
```

- For example:

```
$ dlet -f ultra96v2.hwh
$ mkdir arch/ultra96v2
$ mv dpu-03-26-2020-13-30.dcf arch/ultra96v2/ultra96v2.dcf
```

### Edit ``arch/arch.json``

- Add path the .dcf file path (relative to the model directory of your choice) to ``"dcf"`` section of ``arch/arch.json``

- For example, ``arch/arch.json`` will look like:

```json
{
    "target"   : "dpuv2",
    "dcf"      : "../arch/ultra96v2/ultra96v2.dcf",
    "cpu_arch" : "arm64"
}
```

- Copy edited ``arch.json`` into ``arch/<board name>``

```
$ mv arch.json arch/
```
