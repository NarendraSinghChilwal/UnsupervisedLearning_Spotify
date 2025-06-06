# UnsupervisedLearning_Spotify

This notebook performs K-means clustering on Spotify audio features to generate mood-based playlists. It pulls data directly from the Hugging Face dataset `maharshipandya/spotify-tracks-dataset`.

## Contents

- **notebooks/SpotifyDM.ipynb**  
  - Loads Spotify audio features via:
    ```python
    df = pd.read_csv("hf://datasets/maharshipandya/spotify-tracks-dataset/dataset.csv")
    ```
  - Selects and normalizes key features (e.g., danceability, energy, tempo, etc.).
  - Runs K-means clustering for various _k_, computes silhouette score and Davies–Bouldin index to choose the best _k_.
  - Applies PCA for a 2D visualization of clusters.
  - Visualizes cluster centers and summary statistics.

## Repository Structure

```
UnsupervisedLearning_Spotify/
├── README.md
├── requirements.txt
├── .gitignore
└── notebooks/
    └── SpotifyDM.ipynb
```

## Setup & Usage

1. **Clone the repository**  
   ```bash
   git clone git@github.com:NarendraSinghChilwal/UnsupervisedLearning_Spotify.git
   cd UnsupervisedLearning_Spotify
   ```

2. **Create a virtual environment** (recommended)  
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # (macOS/Linux)
   venv\Scripts\activate      # (Windows)
   ```

3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

4. **Open the notebook**  
   ```bash
   jupyter notebook notebooks/SpotifyDM.ipynb
   ```  
   The notebook will fetch its data directly from Hugging Face at runtime.

## Dependencies

See `requirements.txt` for exact package versions. At minimum, this project uses:

- numpy  
- pandas  
- matplotlib  
- seaborn  
- scikit-learn  

## Notes

- The data is loaded on-the-fly via the Hugging Face URL, so there’s no need to commit a local CSV.  
- If you later extract functions into reusable modules, consider adding a `src/` folder; currently, all code lives in the notebook.

## Contact

narensinghchilwal@gmail.com
