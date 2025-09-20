pip install -r requirements.txt

mkdir -p data/writingPrompts
wget https://dl.fbaipublicfiles.com/fairseq/data/writingPrompts.tar.gz
curl -O https://dl.fbaipublicfiles.com/fairseq/data/writingPrompts.tar.gz
tar -xzf writingPrompts.tar.gz \
    --strip-components=1 \
    -C data/writingPrompts \
    writingPrompts

python scripts/model.py --model_name='gpt-j-6B'
python scripts/model.py --model_name='gpt-neo-2.7B'
python scripts/model.py --model_name='qwen-7b'
python scripts/model.py --model_name='mistralai-7b'
python scripts/model.py --model_name='llama3-8b'
python scripts/model.py --model_name='falcon-7b'
python scripts/model.py --model_name='falcon-7b-instruct'
python scripts/model.py --model_name='gemma-9b'
python scripts/model.py --model_name='gemma-9b-instruct'

NLTK_DIR="$HOME/nltk_data"

# mkdir -p "$NLTK_DIR/corpora"
# mkdir -p "$NLTK_DIR/tokenizers"

# stopwords
wget -O "$NLTK_DIR/stopwords.zip" \
  https://github.com/nltk/nltk_data/raw/refs/heads/gh-pages/packages/corpora/stopwords.zip
unzip -o "$NLTK_DIR/stopwords.zip" -d "$NLTK_DIR/corpora/"
rm "$NLTK_DIR/stopwords.zip"

# wordnet
wget -O "$NLTK_DIR/wordnet.zip" \
  https://github.com/nltk/nltk_data/raw/refs/heads/gh-pages/packages/corpora/wordnet.zip
unzip -o "$NLTK_DIR/wordnet.zip" -d "$NLTK_DIR/corpora/"
rm "$NLTK_DIR/wordnet.zip"

# omw-1.4
wget -O "$NLTK_DIR/omw-1.4.zip" \
  https://github.com/nltk/nltk_data/raw/refs/heads/gh-pages/packages/corpora/omw-1.4.zip
unzip -o "$NLTK_DIR/omw-1.4.zip" -d "$NLTK_DIR/corpora/"
rm "$NLTK_DIR/omw-1.4.zip"

# punkt_tab
wget -O "$NLTK_DIR/punkt_tab.zip" \
  https://github.com/nltk/nltk_data/raw/refs/heads/gh-pages/packages/tokenizers/punkt_tab.zip
unzip -o "$NLTK_DIR/punkt_tab.zip" -d "$NLTK_DIR/tokenizers/"
rm "$NLTK_DIR/punkt_tab.zip"

## Please add NLTK_DATA="$HOME/nltk_data" into the environment path. 