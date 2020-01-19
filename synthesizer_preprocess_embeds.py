from synthesizer_origin.preprocess import create_embeddings
from utils.argutils import print_args
from pathlib import Path
from synthesizer.utils.audio import AudioProcessor
from synthesizer.utils.generic_utils import load_config
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Creates embeddings for the synthesizer from the LibriSpeech utterances.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("synthesizer_root", type=Path, help=\
        "Path to the synthesizer training data that contains the audios and the train.txt file. "
        "If you let everything as default, it should be <datasets_root>/SV2TTS/synthesizer/.")
#     parser.add_argument(
#         '-c',
#         '--config_path',
#         type=str,
#         help='Path to config file for training.',
#         required=True
#     )
    parser.add_argument("-e", "--encoder_model_fpath", type=Path, 
                        default="encoder/saved_models/pretrained.pt", help=\
        "Path your trained encoder model.")
    parser.add_argument("-n", "--n_processes", type=int, default=4, help= \
        "Number of parallel processes. An encoder is created for each, so you may need to lower "
        "this value on GPUs with low memory. Set it to 1 if CUDA is unhappy.")
    args = parser.parse_args()
    
    # Preprocess the dataset
    print_args(args, parser)
    # c = load_config(args.config_path)
    # ap = AudioProcessor(**c.audio)
    create_embeddings(**vars(args))    
