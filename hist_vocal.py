"""import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the file and parse data into a DataFrame
file_path = "/home/usuaris/veussd/DATABASES/Ocean/llista_paths_audios/spectrograms_5_sec_new.lst"

data = []
with open(file_path, "r") as file:
    for line in file:
        parts = line.strip().split("_")
        species = parts[2]
        vocalization = parts[3]
        data.append({"Species": species, "Vocalization": vocalization})

df = pd.DataFrame(data)

# Create histograms for both species
blue_data = df[df["Species"] == "Blue"]
fin_data = df[df["Species"] == "Fin"]

# Set a custom style for the plots
sns.set_style("whitegrid")
sns.set_palette("pastel")

# Plot and save the Blue Whale Vocalizations histogram
plt.figure(figsize=(10, 6))
sns.countplot(data=blue_data, x="Vocalization")
plt.title("Blue Whale Vocalizations")
plt.xlabel("Vocalization")
plt.ylabel("Number of Samples")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("blue_whale_vocalizations.png")

# Plot and save the Fin Whale Vocalizations histogram
plt.figure(figsize=(10, 6))
sns.countplot(data=fin_data, x="Vocalization")
plt.title("Fin Whale Vocalizations")
plt.xlabel("Vocalization")
plt.ylabel("Number of Samples")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("fin_whale_vocalizations.png")

plt.show()
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Data for different whale species and number of samples
species_data = {
    "Species": ["Blue", "Fin", "Humpback", "Minke", "Unidentified"],
    "Number of Samples": [48834, 27193, 208, 1495, 28138]
}

df = pd.DataFrame(species_data)

# Set a custom style for the plot
sns.set_style("whitegrid")
sns.set_palette("pastel")

# Plot and save the histogram
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x="Species", y="Number of Samples")
plt.title("Number of Samples for Different Whale Species")
plt.xlabel("Species")
plt.ylabel("Number of Samples")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("whale_species_histogram.png")

plt.show()
