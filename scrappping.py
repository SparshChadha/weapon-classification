import requests
from bs4 import BeautifulSoup
import os

def imagedown(queries, dest_folder, num_images):
    for query in queries:
        # Create the search URL
        url = f"https://www.google.com/search?hl=en&tbm=isch&q={query.replace(' ', '+')}"

        # Create the folder for the current query
        folder_path = os.path.join(dest_folder, query.replace(' ', '_'))
        os.makedirs(folder_path, exist_ok=True)

        r = requests.get(url)
        soup = BeautifulSoup(r.text, 'html.parser')
        
        # Find all image tags
        images = soup.find_all('img')
        
        for idx, image in enumerate(images[:num_images]):  # Limit the number of images
            try:
                link = image['src']
                img_name = f"{query.replace(' ', '_')}_{idx + 1}.jpg"
                img_path = os.path.join(folder_path, img_name)
                with open(img_path, 'wb') as f:
                    im = requests.get(link)
                    f.write(im.content)
                    print(f'Writing: {img_path}')
            except Exception as e:
                print(f"Could not download image {idx + 1}: {e}")

if __name__ == "__main__":
    queries = [
    "Pistols", "Revolvers", "Assault Rifles", "Sniper Rifles", "Shotguns", 
    "Submachine Guns", "Machine Guns", "Knives", "Machetes", "Daggers", 
    "Hatchets", "Battle Axes", "Hand Grenades", "Flashbangs", "Bomb", 
    "Molotov Cocktails", "Katana", "Sabre", "Throwing Knives", "Bayonets", 
    "Kukri", "Tomahawks", "Revolver Shotguns", "Machine Pistols", 
    "Bolt-Action Rifles", "Lever-Action Rifles", "Semi-Automatic Rifles", 
    "Smoke Grenades", "Stun Grenades", "Chemical Bombs", "Thermite Charges", 
    "Crossbows", "Slingshots", "Blowguns", "Batons", "Pepper Spray", 
    "Tasers",

    # Non-Weapon Items
    "Hammers", "Screwdrivers", "Wrenches", "Pliers", "Saws", "Drills", 
    "Crowbars", "Kitchen Knives", "Scissors", "Box Cutters", "Forks", 
    "Spoons", "Gardening Tools", "Pens", "Pencils", "Rulers", 
    "Nail Clippers", "Toy Guns", "Toy Swords", "Toy Knives", 
    "Plastic Tools", "Action Figures", "Toy Grenades", "Toy Axes", 
    "Baseball Bats", "Hockey Sticks", "Golf Clubs", "Archery Bows", 
    "Fishing Knives", "Fencing Swords", "Remote Controls", "Smartphones", 
    "Laptops", "Tablets", "Flashlights", "Power Banks", "Headphones", 
    "Umbrellas", "Walking Canes", "Camera Tripods", "Flash Drives", 
    "Sunglasses", "Wallets", "Keychains", "Car Keys", "Water Bottles", 
    "Pipes", "Bolts", "Tape Measures", "Spirit Levels", "Chains", 
    "Construction Helmets"
]
  # Example array of search queries
    dest_folder = "/Users/sparsh/research paper/images"  # Destination folder
    num_images = 500  # Number of images to download per query
    imagedown(queries, dest_folder, num_images)
