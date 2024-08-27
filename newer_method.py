import pandas as pd
import nltk
import re
import json
from nltk.tokenize import word_tokenize

# Download NLTK resources
nltk.download('punkt')

# Load data
data = pd.read_csv('FinalLabel_2.csv', delimiter=';')

# Mengonversi kolom Date ke datetime
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')


# Read abbreviation mappings from a text file
def load_abbreviation_mappings(filepath):
    with open(filepath, 'r') as file:
        abbreviation_mapping = json.load(file)
    return abbreviation_mapping


# Load abbreviation mapping
abbreviation_mapping = load_abbreviation_mappings('combined_slang_words.txt')


# Function to replace abbreviations with their full form
def replace_abbreviations(text, abbreviation_mapping):
    tokens = word_tokenize(text)

    new_tokens = []
    for token in tokens:
        split_tokens = re.split(r'[^\w\s]', token)
        cleaned_tokens = [re.sub(r'\d+', '', word) for word in split_tokens]
        new_tokens.extend(cleaned_tokens)
    tokens = new_tokens

    replaced_tokens = [abbreviation_mapping.get(word, word) for word in tokens]
    return ' '.join(replaced_tokens)


# Fungsi-fungsi untuk memeriksa kriteria ulasan palsu
def is_untruthful_opinion(review_text, rating):
    # Daftar kata kunci positif dan negatif
    positive_keywords = ["terbaik", "bagus", "baik", "sempurna", "luar biasa", "hebat",
                         "fantastis", "worth", "worth it", "rekomended", "rekomen", "recommed", "recommended"]
    negative_keywords = ["terburuk", "buruk", "tidak bagus", "jelek", "rusak",
                         "hancur", "parah", "kacau", "mengecewakan", "kecewa"]
    ambiguous_phrases = ["belum di coba", "gatau", "tidak tahu", "belum dicoba", "nggak tau",
                         "belum coba", "masih dicoba", "belum yakin", "belum test", "belum dipakai", "nanti dicoba"]

    review_text_lower = review_text.lower()

    # Cek apakah review positif tapi rating jelek (≤ 2)
    if any(word in review_text_lower for word in positive_keywords) and rating <= 2:
        return True
    # Cek apakah review negatif tapi rating tinggi (≥ 4)
    elif any(word in review_text_lower for word in negative_keywords) and rating >= 4:
        return True

    # Cek apakah ulasan ambigu atau tidak jelas
    if any(phrase in review_text_lower for phrase in ambiguous_phrases):
        return True

    return False


def is_review_on_brand_only(review_text):
    brand_keywords = ["merek", "brand", "perusahaan", "merk", "toko", "store",
                      "distributor", "penjual", "supplier", "retailer", "pabrik", "vendor", "importir"]
    return any(word in review_text.lower() for word in brand_keywords)


def is_non_review(review_text):
    non_review_keywords = ["pengiriman", "pengemasan", "kemasan", "layanan", "pelayanan", "packing",
                           "packaging", "kurir", "service", "customer service", "respon", "respons", "penanganan"]
    return any(word in review_text.lower() for word in non_review_keywords)


def is_review_length_short(review_text, min_length=60, max_length=200):
    return len(review_text) < min_length or len(review_text) > max_length


def is_burstiness(review_dates, username, threshold=2):
    if len(review_dates) < threshold:
        return False

    # Sort the review_dates by date to ensure they are in chronological order
    review_dates.sort()

    for i in range(len(review_dates) - 1):
        time_diff = (review_dates[i + 1] -
                     review_dates[i]).total_seconds() / 3600
        if time_diff < 24:
            return True

    return False


def ngram(text, n=3):
    """Membuat n-gram dari teks."""
    text = text.lower()
    return {text[i:i+n] for i in range(len(text) - n + 1)}


def jaccard_similarity(set1, set2):
    """Menghitung Jaccard similarity antara dua set n-gram."""
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union) if union else 0


def is_maximum_content_similarity(review_text, other_reviews, n=2, similarity_threshold=0.5):
    """Memeriksa apakah kemiripan maksimum berdasarkan Similarity N-Gram Jaccard melebihi threshold."""
    review_ngrams = ngram(review_text, n)
    max_similarity = 0

    for other_review in other_reviews:
        other_review_ngrams = ngram(other_review, n)
        similarity = jaccard_similarity(review_ngrams, other_review_ngrams)

        if similarity > max_similarity:
            max_similarity = similarity

        if max_similarity > similarity_threshold:
            return True

    return False


def is_extreme_rating(rating):
    return rating == 1 or rating == 5


# Fungsi pelabelan yang memberikan hasil per kriteria
def label_reviews_per_criteria(df):
    labeled_reviews = []
    previous_reviews = {}

    for index, row in df.iterrows():
        review_text = row["Review"]
        review_date = row["Date"]
        rating = row["Rating"]
        reviewer_id = row["Username"]
        other_reviews = df[df['Username'] != reviewer_id]['Review'].tolist()
        review_dates = df[df['Username'] == reviewer_id]['Date'].tolist()

        # Lakukan lowercasing dan ganti singkatan sebelum pelabelan
        processed_review = replace_abbreviations(
            review_text.lower(), abbreviation_mapping)

        # Print informasi tentang ulasan yang sedang diproses
        print(
            f"Memproses ulasan ke-{index + 1} dari {len(df)}: Reviewer - {reviewer_id}, Rating - {rating}")

        # Memeriksa setiap kriteria
        labels = {
            "Untruthful Opinion": is_untruthful_opinion(processed_review, rating),
            "Review on Brand Only": is_review_on_brand_only(processed_review),
            "Non-Review": is_non_review(processed_review),
            "Review Length Short": is_review_length_short(processed_review),
            "Burstiness": is_burstiness(review_dates, reviewer_id),
            "Maximum Content Similarity": is_maximum_content_similarity(processed_review, other_reviews),
            "Extreme Rating": is_extreme_rating(rating)
        }

        # Simpan hasil label per kriteria
        result = {
            "Username": reviewer_id,
            "Review": review_text,  # tetap simpan review original
            **{key: ("Palsu" if value else "Asli") for key, value in labels.items()}
        }
        labeled_reviews.append(result)
        previous_reviews[reviewer_id] = processed_review

    return pd.DataFrame(labeled_reviews)


# Melabelkan ulasan menggunakan DataFrame
labeled_reviews_df = label_reviews_per_criteria(data)

# Ganti dengan nama file output Anda
output_file = "labeled_reviews_per_criteria.csv.csv"
labeled_reviews_df.to_csv(output_file, index=False, sep=';')

print(f"Hasil pelabelan disimpan ke {output_file}")
