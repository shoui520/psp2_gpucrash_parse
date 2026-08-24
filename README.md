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

Other basic commands:

```sh
./psp2-gpucrash-parse registers dump.psp2dmp
./psp2-gpucrash-parse segments dump.psp2dmp
./psp2-gpucrash-parse tty dump.psp2dmp
./psp2-gpucrash-parse compare first.psp2dmp second.psp2dmp
```
