# Subtitle Data

This directory is intended for storing subtitle files (.srt) used by the search engine.

## How to add data

1. Download the subtitle files from the provided Google Drive link:
   https://drive.google.com/drive/folders/1ZJtMu05v2QcFsL1M8y5NMWJFQ2bZCljC?usp=drive_link

2. Place all .srt files directly in this directory.

3. The search engine will automatically detect and load these files.

## Data Format

The system expects subtitle files in the SRT format, which typically looks like:

```
1
00:00:20,000 --> 00:00:24,400
This is the first subtitle text.

2
00:00:24,600 --> 00:00:27,800
This is the second subtitle text.
```

## Notes

- Large datasets may require significant processing time and memory.
- Consider using a sample of the data for initial testing (use the `sample_size` parameter).
- The search engine will automatically preprocess these files to remove timestamps and clean the text.
