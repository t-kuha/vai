# vai: Vitis AI models

- Make sure to create Vitis AI-enabled platform & build a [Reference Design](https://github.com/Xilinx/Vitis-AI/tree/master/dpu)

***

## Preparation for Vitis AI v1.3

- Copy ``arch.json`` in ``sd_card`` directory of DPU-TRD as ``arch/<platform name>.json``

***

## Preparation for Vitis AI v1.2 and before

<details>
  <summary>click here</summary>

### Generate .dcf file

```shell-session
$ dlet -f <input .hwh file>
$ mkdir arch/<platform name>
$ mv <generated file> arch/<platform name>/<platform name>.dcf
```

- For example:

```shell-session
$ dlet -f ultra96v2.hwh
$ mkdir arch/ultra96v2
$ mv dpu-03-26-2020-13-30.dcf arch/ultra96v2/ultra96v2.dcf
```

### Edit arch/arch.json

- Add path the .dcf file path (relative to the model directory of your choice) to ``"dcf"`` section of ``arch/arch.json``

- For example, ``arch/arch.json`` will look like:

```json
{
    "target"   : "dpuv2",
    "dcf"      : "../arch/ultra96v2/ultra96v2.dcf",
    "cpu_arch" : "arm64"
}
```

- Copy edited ``arch.json`` into ``arch/<platform name>``

```shell-session
$ mv arch/arch.json arch/<platform name>
```
</details>
