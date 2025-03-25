# Total cards in a deck
total_cards = 52

# Probability of drawing a Queen
P_Queen = 4 / total_cards  # 4 Queens in the deck

# Probability of drawing a Heart
P_Heart = 13 / total_cards  # 13 Hearts in the deck

# Probability of drawing the Queen of Hearts (overlap)
P_Queen_and_Heart = 1 / total_cards  # 1 Queen of Hearts

# Using the non-mutual exclusive addition rule:
P_Queen_or_Heart = P_Queen + P_Heart - P_Queen_and_Heart

# Print results
print(f"### Probability of Drawing a Queen or a Heart ###\n")
print(f"P(Queen) = {P_Queen:.4f}")
print(f"P(Heart) = {P_Heart:.4f}")
print(f"P(Queen and Heart) = {P_Queen_and_Heart:.4f}")
print(f"\nP(Queen or Heart) = P(Queen) + P(Heart) - P(Queen and Heart)")
print(f"P(Queen or Heart) = {P_Queen:.4f} + {P_Heart:.4f} - {P_Queen_and_Heart:.4f}")
print(f"P(Queen or Heart) = {P_Queen_or_Heart:.4f} (or 4/13 ≈ 0.3077)")

print("\nThis calculation accounts for the overlap of the Queen of Hearts.")
