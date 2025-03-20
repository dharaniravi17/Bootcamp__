import hashlib

def verify_sha256(file_path, given_hash):
    try:
        # Read the file and compute its SHA256 hash
        with open(file_path, "rb") as file:
            file_data = file.read()
        computed_hash = hashlib.sha256(file_data).hexdigest()

        # Compare hashes
        if computed_hash == given_hash:
            print("✅ File is authentic! Hashes match.")
        else:
            print("❌ WARNING: File has been modified! Hashes do NOT match.")
    
    except Exception as e:
        print(f"❌ Error: {e}")

# Example Usage
file_path = input("📂 Enter file path: ").strip()
given_hash = input("🔑 Enter stored SHA256 hash: ").strip()

verify_sha256(file_path, given_hash)
