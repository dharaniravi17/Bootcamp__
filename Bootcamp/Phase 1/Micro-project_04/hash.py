import hashlib

# Step 1: Get user input
text = input("Enter the text to encrypt: ")

# Step 2: Save the input text to "message.txt"
with open("message.txt", "w") as file:
    file.write(text)

# Step 3: Read the content from the file
with open("message.txt", "r") as file:
    content = file.read()

# Step 4: Encode and hash the content
hash_object = hashlib.sha256(content.encode())
hash_value = hash_object.hexdigest()

# Step 5: Save the hash to "encrypted.txt"
with open("encrypted.txt", "w") as file:
    file.write(hash_value)

# Step 6: Print the hash for verification
print(f"\n🔹 SHA256 Hash:\n{hash_value}")
print("✅ Hash saved in 'encrypted.txt'")
