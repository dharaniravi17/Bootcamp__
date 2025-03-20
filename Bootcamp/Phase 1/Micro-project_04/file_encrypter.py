import hashlib
import os

# Function to generate SHA256 hash of any file
def hash_file(file_path):
    try:
        # Ensure the file exists
        if not os.path.isfile(file_path):
            print("❌ Error: File not found!")
            return
        
        # Read the file in binary mode
        with open(file_path, "rb") as file:
            file_data = file.read()
        
        # Generate SHA256 hash
        hash_object = hashlib.sha256(file_data)
        hash_value = hash_object.hexdigest()
        
        # Save hash to a .hash file with the same name
        output_file = file_path + ".hash"
        with open(output_file, "w") as file:
            file.write(hash_value)

        print(f"✅ File '{file_path}' encrypted successfully!")
        print(f"🔹 SHA256 Hash: {hash_value}")
        print(f"🔹 Hash saved in: {output_file}")

    except Exception as e:
        print(f"❌ An error occurred: {e}")

# Get file path input from the user
file_path = input("📂 Enter the file path to encrypt: ").strip()

# Encrypt the file
hash_file(file_path)
