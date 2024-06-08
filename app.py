import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import json
import os
import torchvision.models as models
import requests

# Define the label map
label_map = {
    "0": "Akame ga Kill!",
    "1": "Angel Beats!",
    "2": "Ano Hi Mita Hana no Namae wo Bokutachi wa Mada Shiranai",
    "3": "Another",
    "4": "Ansatsu Kyoushitsu",
    "5": "Ao no Exorcist",
    "6": "BANANA FISH",
    "7": "Bakemonogatari",
    "8": "Black Clover",
    "9": "Bleach",
    "10": "Boku dake ga Inai Machi",
    "11": "Boku no Hero Academia",
    "12": "Bungou Stray Dogs",
    "13": "CLANNAD",
    "14": "Chainsaw Man",
    "15": "Charlotte",
    "16": "Chuunibyou demo Koi ga Shitai",
    "17": "Code Geass_ Hangyaku no Lelouch",
    "18": "Cowboy Bebop",
    "19": "Darling in the Franxx",
    "20": "Death Note",
    "21": "Death Parade",
    "22": "Devilman Crybaby",
    "23": "Dororo",
    "24": "Dr. Stone",
    "25": "Dungeon ni Deai wo Motomeru no wa Machigatteiru Darou ka",
    "26": "Durarara!!",
    "27": "Enen no Shouboutai",
    "28": "Fairy Tail",
    "29": "Fate Zero",
    "30": "Fruits Basket_ 1st Season",
    "31": "Fumetsu no Anata e",
    "32": "Go-toubun no Hanayome",
    "33": "Goblin Slayer",
    "34": "Hagane no Renkinjutsushi_ Fullmetal Alchemist",
    "35": "Haikyuu!!",
    "36": "Hataraku Maou-sama!",
    "37": "High School DxD",
    "38": "Horimiya",
    "39": "Howl no Ugoku Shiro",
    "40": "Hunter x Hunter (2011)",
    "41": "Hyouka",
    "42": "JoJo",
    "43": "Jujutsu Kaisen",
    "44": "Kaguya-sama wa Kokurasetai_ Tensaitachi no Renai Zunousen",
    "45": "Kakegurui",
    "46": "Kami no Tou_ Tower of God",
    "47": "Kanojo, Okarishimasu",
    "48": "Kill la Kill",
    "49": "Kimetsu no Yaiba",
    "50": "Kimi no Na wa.",
    "51": "Kimi no Suizou wo Tabetai",
    "52": "Kiseijuu_ Sei no Kakuritsu",
    "53": "Kobayashi-san Chi no Maidragon",
    "54": "Koe no Katachi",
    "55": "Komi-san wa, Komyushou desu",
    "56": "Kono Subarashii Sekai ni Shukufuku wo!",
    "57": "Made in Abyss",
    "58": "Mahou Shoujo Madoka Magica",
    "59": "Mirai Nikki",
    "60": "Mob Psycho 100",
    "61": "Monster",
    "62": "Mushoku Tensei_ Isekai Ittara Honki Dasu",
    "63": "Nanatsu no Taizai",
    "64": "Naruto",
    "65": "No Game No Life",
    "66": "Noragami",
    "67": "One Piece",
    "68": "One Punch Man",
    "69": "Overlord",
    "70": "Owari no Seraph",
    "71": "Psycho-Pass",
    "72": "Re_Zero kara Hajimeru Isekai Seikatsu",
    "73": "Saiki Kusuo no Psi-nan",
    "74": "Seishun Buta Yarou wa Bunny Girl Senpai no Yume wo Minai",
    "75": "Sen to Chihiro no Kamikakushi",
    "76": "Shigatsu wa Kimi no Uso",
    "77": "Shin Seiki Evangelion",
    "78": "Shingeki no Kyojin",
    "79": "Shokugeki no Souma",
    "80": "Sono Bisque Doll wa Koi wo Suru",
    "81": "Soul Eater",
    "82": "Spy x Family",
    "83": "Steins Gate",
    "84": "Sword Art Online",
    "85": "Tate no Yuusha no Nariagari",
    "86": "Tengen Toppa Gurren Lagann",
    "87": "Tenki no Ko",
    "88": "Tensei Shitara Slime Datta Ken",
    "89": "The God of High School",
    "90": "Tokyo Ghoul",
    "91": "Tokyo Revengers",
    "92": "Toradora!",
    "93": "Vinland Saga",
    "94": "Violet Evergarden",
    "95": "Wotaku ni Koi wa Muzukashii",
    "96": "Yahari Ore no Seishun Love Come wa Machigatteiru",
    "97": "Yakusoku no Neverland",
    "98": "Youkoso Jitsuryoku Shijou Shugi no Kyoushitsu e",
    "99": "Sousou no Frieren"
}


# Save the label map to a JSON file
label_map_path = 'label_map.json'
with open(label_map_path, 'w') as f:
    json.dump(label_map, f)

# Define the model
class MyResNeXt101(nn.Module):
    def __init__(self, num_classes=100):
        super(MyResNeXt101, self).__init__()
        self.network = models.resnext101_32x8d(pretrained=True)

        in_features = self.network.fc.in_features
        self.network.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.network(x)

# Load the model
model = MyResNeXt101(num_classes=100)
state_dict = torch.load('.//best_model Resnext101.pth', map_location=torch.device('cpu'))
model.load_state_dict(state_dict)
model.eval()

# Define the test transformations
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load or create the label map
if os.path.exists(label_map_path):
    with open(label_map_path) as f:
        label_map = json.load(f)

# Preprocess function
def preprocess_image(image):
    # Open the image and convert it to RGB format
    image = Image.open(image).convert('RGB')

    # If the image is in WEBP format, convert it to RGB
    if image.format == 'WEBP':
        # Convert the image to RGB format
        image = image.convert('RGB')
    
    # Apply the transformations
    image = test_transforms(image)
    
    # Add a dimension to the image tensor
    image = image.unsqueeze(0)
    
    return image

# Prediction function
def predict(image):
    image_tensor = preprocess_image(image)
    with torch.no_grad():
        outputs = model(image_tensor)
    probabilities = nn.functional.softmax(outputs, dim=1)[0]
    top5_prob, top5_labels = torch.topk(probabilities, 5)
    top5_labels = [label_map[str(label.item())] for label in top5_labels]
    top5_prob = [round(prob.item() * 100, 2) for prob in top5_prob]
    return top5_labels, top5_prob

anilist_id_match = {
        "Akame ga Kill!": ["20613"],
        "Angel Beats!": ["6547", "9062", "10067"],
        "Ano Hi Mita Hana no Namae wo Bokutachi wa Mada Shiranai": ["9989", "15039", "107342"],
        "Another": ["11111"],
        "Ansatsu Kyoushitsu": ["20755", "21170"],
        "Ao no Exorcist": ["9919"],
        "BANANA FISH": ["100388"],
        "Bakemonogatari": ["5081"],
        "Black Clover": ["97940"],
        "Bleach": ["269", "159322"],
        "Boku dake ga Inai Machi": ["21234"],
        "Boku no Hero Academia": ["21459", "21856", "100166", "104276", "117193", "139630", "163139"],
        "Bungou Stray Dogs": ["21311", "21679", "103223", "141249", "163263"],
        "CLANNAD": ["2167"],
        "Chainsaw Man": ["127230"],
        "Charlotte": ["20997"],
        "Chuunibyou demo Koi ga Shitai": ["14741", "18671"],
        "Code Geass_ Hangyaku no Lelouch": ["1575", "2904"],
        "Cowboy Bebop": ["1"],
        "Darling in the Franxx": ["99423"],
        "Death Note": ["1535"],
        "Death Parade": ["20931"],
        "Devilman Crybaby": ["98460"],
        "Dororo": ["101347"],
        "Dr. Stone": ["105333", "113936", "131518", "162670"],
        "Dungeon ni Deai wo Motomeru no wa Machigatteiru Darou ka": ["20920", "101167", "112124"],
        "Durarara!!": ["6746", "20652", "20879", "20880"],
        "Enen no Shouboutai": ["105310", "114236"],
        "Fairy Tail": ["6702", "99749", "20626"],
        "Fate Zero": ["10087", "11741"],
        "Fruits Basket_ 1st Season": ["105334"],
        "Fumetsu no Anata e": ["114535", "138565"],
        "Go-toubun no Hanayome": ["103572", "163327", "109261", "131520"],
        "Goblin Slayer": ["101165", "129188"],
        "Hagane no Renkinjutsushi_ Fullmetal Alchemist": ["5114"],
        "Haikyuu!!": ["20464"],
        "Hataraku Maou-sama!": ["15809", "130592", "155168"],
        "High School DxD": ["11617", "15451", "20745", "97767"],
        "Horimiya": ["124080", "163132"],
        "Howl no Ugoku Shiro": ["431"],
        "Hunter x Hunter (2011)": ["11061"],
        "Hyouka": ["12189"],
        "JoJo": ["14719", "20474", "20799", "21450", "131942", "146722"],
        "Jujutsu Kaisen": ["113415", "145064", "131573"],
        "Kaguya-sama wa Kokurasetai_ Tensaitachi no Renai Zunousen": ["101921", "112641", "125367"],
        "Kakegurui": ["98314", "100876"],
        "Kami no Tou_ Tower of God": ["115230"],
        "Kanojo, Okarishimasu": ["113813", "124410", "154745"],
        "Kill la Kill": ["18679"],
        "Kimetsu no Yaiba": ["101922", "142329", "145139", "166240"],
        "Kimi no Na wa.": ["21519"],
        "Kimi no Suizou wo Tabetai": ["99750"],
        "Kiseijuu_ Sei no Kakuritsu": ["20623"],
        "Kobayashi-san Chi no Maidragon": ["21776", "107717", "132096"],
        "Koe no Katachi": ["20954"],
        "Komi-san wa, Komyushou desu": ["133965", "142984"],
        "Kono Subarashii Sekai ni Shukufuku wo!": ["21202", "21699", "136804"],
        "Made in Abyss": ["97986", "114745"],
        "Mahou Shoujo Madoka Magica": ["9756", "104051"],
        "Mirai Nikki": ["10620"],
        "Mob Psycho 100": ["21507", "101338", "140439"],
        "Monster": ["19"],
        "Mushoku Tensei_ Isekai Ittara Honki Dasu": ["108465", "127720", "146065", "166873"],
        "Nanatsu no Taizai": ["20789", "99539", "108928", "116752", "148862"],
        "Naruto": ["20", "1735"],
        "No Game No Life": ["19815"],
        "Noragami": ["20447", "21128"],
        "One Piece": ["21"],
        "One Punch Man": ["21087", "97668"],
        "Overlord": ["20832", "98437", "101474", "133844"],
        "Owari no Seraph": ["20829", "21483", "20993"],
        "Psycho-Pass": ["13601", "20513", "108307"],
        "Re_Zero kara Hajimeru Isekai Seikatsu": ["21355", "108632", "119661"],
        "Saiki Kusuo no Psi-nan": ["21804", "98034"],
        "Seishun Buta Yarou wa Bunny Girl Senpai no Yume wo Minai": ["101291"],
        "Sen to Chihiro no Kamikakushi": ["199"],
        "Shigatsu wa Kimi no Uso": ["20665"],
        "Shin Seiki Evangelion": ["30"],
        "Shingeki no Kyojin": ["16498", "110277", "20958", "99147", "131681", "104578", "162314"],
        "Shokugeki no Souma": ["20923", "21518", "99255", "109963", "114043", "100773"],
        "Sono Bisque Doll wa Koi wo Suru": ["132405"],
        "Sousou no Frieren": ["154587"],
        "Soul Eater": ["3588"],
        "Spy x Family": ["140960", "142838", "158927"],
        "Steins Gate": ["9253", "21127"],
        "Sword Art Online": ["11757", "20594", "100183", "108759"],
        "Tate no Yuusha no Nariagari": ["99263", "111321", "111322"],
        "Tengen Toppa Gurren Lagann": ["2001"],
        "Tenki no Ko": ["106286"],
        "Tensei Shitara Slime Datta Ken": ["101280", "108511", "156822", "116742"],
        "The God of High School": ["116006"],
        "Tokyo Ghoul": ["20605", "20850", "100240", "102351"],
        "Tokyo Revengers": ["120120", "142853", "163329"],
        "Toradora!": ["4224"],
        "Vinland Saga": ["101348", "136430"],
        "Violet Evergarden": ["21827"],
        "Wotaku ni Koi wa Muzukashii": ["99578"],
        "Yahari Ore no Seishun Love Come wa Machigatteiru": ["14813", "108489", "20698"],
        "Yakusoku no Neverland": ["101759", "108725"],
        "Youkoso Jitsuryoku Shijou Shugi no Kyoushitsu e": ["98659", "145545", "146066"]
    }

def fetch_anilist_info(anilist_id):
    url = f"https://graphql.anilist.co"
    query = """
    query ($id: Int) {
        Media(id: $id, type: ANIME) {
            averageScore
            popularity
            genres
            tags {
                name
            }
        }
    }
    """
    variables = {"id": anilist_id}
    response = requests.post(url, json={"query": query, "variables": variables})
    data = response.json()
    return data.get("data", {}).get("Media", {})

# Function to format large numbers with commas
def format_number_with_commas(number):
    return f"{number:,}"

st.title('AniRec 🔎 (Anime Recognition System from Scene Images)')
st.write('Upload an anime scene image to identify it.')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("Classifying...")
    top5_labels, top5_prob = predict(uploaded_file)
    st.write("<h3>Top 5 Predictions:</h3>", unsafe_allow_html=True)
    
    max_label_length = max(len(label) for label in top5_labels)
    for i, (label, prob) in enumerate(zip(top5_labels, top5_prob), 1):
        underlined_label = f'<u>{i}) {label}</u> : {prob}%'
        padding_spaces = ' ' * (max_label_length - len(label))
        st.markdown(f'{underlined_label}:{padding_spaces}', unsafe_allow_html=True)
        
        # Display the image corresponding to the label
        label_image_path_jpg = f'images/{label}.jpg'  # Assuming images are stored in a directory named 'images'
        label_image_path_png = f'images/{label}.png'  # Assuming images are stored in a directory named 'images'

        if os.path.exists(label_image_path_jpg):
            label_image = Image.open(label_image_path_jpg)
            st.image(label_image, caption=label, width=150)
        elif os.path.exists(label_image_path_png):
            label_image = Image.open(label_image_path_png)
            st.image(label_image, caption=label, width=150)
        else:
            st.write(f"Image not found for label : {label}")

        # Fetch AniList info using the first AniList ID
        anilist_ids = anilist_id_match.get(label, [])
        if anilist_ids:
            anilist_id = int(anilist_ids[0])
            anilist_info = fetch_anilist_info(anilist_id)
            average_score = anilist_info.get("averageScore", "N/A")
            popularity = anilist_info.get("popularity", "N/A")
            if popularity != "N/A":
                popularity = format_number_with_commas(popularity)
            genres = ", ".join(anilist_info.get("genres", []))
            tags = ", ".join([tag["name"] for tag in anilist_info.get("tags", [])[:5]])  # Extracting only the first 5 tags
            st.write(f"<h6>AniList Information for {label} : </h6>", unsafe_allow_html=True)  # Bigger text for AniList Information
            st.write(f"Average Score : {average_score}")
            st.write(f"Popularity : {popularity}")
            st.write(f"Genres : {genres}")
            st.write(f"Tags : {tags}")
