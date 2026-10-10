# psp2-gpucrash-parse

Parse a GPUCRASH dump:

```sh
./psp2-gpucrash-parse analyze dump.psp2dmp
```

Use an original Vita ELF or SELF for additional CPU address analysis:

```sh
./psp2-gpucrash-parse analyze dump.psp2dmp \
  --image application.self \
  --sdk /path/to/vitasdk
```

Write the complete analysis as JSON:

```sh
./psp2-gpucrash-parse analyze dump.psp2dmp --json
```

GPUCRASH analysis does not require a devkit sidecar. If one is available:

```sh
./psp2-gpucrash-parse analyze dump.psp2dmp --gpu-dump gpu-dump.bin
./psp2-gpucrash-parse gpu-dump gpu-dump.bin --core-dump dump.psp2dmp
./psp2-gpucrash-parse gpu-dump gpu-dump.bin --text
```

Other basic commands:

```sh
./psp2-gpucrash-parse address dump.psp2dmp 0x70000000
./psp2-gpucrash-parse xref dump.psp2dmp 0x70000000
./psp2-gpucrash-parse map dump.psp2dmp
./psp2-gpucrash-parse modules dump.psp2dmp
./psp2-gpucrash-parse module dump.psp2dmp SceGxm
./psp2-gpucrash-parse threads dump.psp2dmp
./psp2-gpucrash-parse thread dump.psp2dmp 0x40010001
./psp2-gpucrash-parse waits dump.psp2dmp
./psp2-gpucrash-parse object dump.psp2dmp 0x40010002
./psp2-gpucrash-parse search dump.psp2dmp --pointers-to 0x70000000
./psp2-gpucrash-parse gpu faults dump.psp2dmp
./psp2-gpucrash-parse gpu contexts dump.psp2dmp
./psp2-gpucrash-parse gpu pipeline dump.psp2dmp
./psp2-gpucrash-parse gpu resources dump.psp2dmp
./psp2-gpucrash-parse registers dump.psp2dmp
./psp2-gpucrash-parse segments dump.psp2dmp
./psp2-gpucrash-parse tty dump.psp2dmp
./psp2-gpucrash-parse validate dump.psp2dmp
./psp2-gpucrash-parse compare first.psp2dmp second.psp2dmp
./psp2-gpucrash-parse triage *.psp2dmp
```

Add `--json` to query commands for machine-readable output.
