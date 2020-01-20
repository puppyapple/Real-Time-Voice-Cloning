# WaveRNN

This code is forked from https://github.com/fatchord/WaveRNN and optimized for [Mozilla-TTS](https://github.com/mozilla/TTS).

# Released Models
| Models        |Dataset | Commit            | Audio Sample  | TTS model | Details|
| ------------- |:------:|:-----------------:|:--------------|:--------|--------- |
| [mold model](https://drive.google.com/drive/folders/1wpPn3a0KQc6EYtKL0qOi4NqEmhML71Ve?usp=sharing) |LJspeech| 8a1c152 | [soundcloud](https://soundcloud.com/user-565970875/ljspeech-logistic-wavernn) | [Tacotron2-iter-260K](https://drive.google.com/drive/folders/1FJRjGDAqWIyZRX4CsppaIPEW8UWXCWzF) | Model with Mixture of Logistic Distribution |
| [10bit model](https://drive.google.com/drive/folders/1VnTJfg2zmvochFNyX7oyUv9TFq6JsnVp?usp=sharing) | LJSpeech | faea90b | [soundcloud](https://soundcloud.com/user-565970875/commonvoice-1) | [Tacotron2-iter-260K](https://drive.google.com/drive/folders/1FJRjGDAqWIyZRX4CsppaIPEW8UWXCWzF) | 10bit Softmax output |
| [universal vocoder](https://drive.google.com/drive/u/1/folders/15JhAbc91dT-RRZwakh_v4tBVOEuoOikg) | LibriTTS | 12c8744 | [soundcloud](https://soundcloud.com/user-565970875/sets/universal-vocoder-with-wavernn) |  - | [(details)](https://github.com/mozilla/TTS/issues/221) |

Check this [TTS notebook](https://github.com/mozilla/TTS/blob/master/notebooks/Benchmark.ipynb) to see TTS+WaveRNN in action. 
To train your own model, you can use [ExtractTTSSpectrogram](https://github.com/erogol/WaveRNN/blob/master/notebooks/ExtractTTSpectrogram.ipynb) to generate spectrograms by TTS and train WaveRNN.
It might be also interesting to check this [TTS issue](https://github.com/mozilla/TTS/issues/26) to catchup with the current state.
