import os

# # Check if data directory exists
# print(os.getcwd())
# print(os.path.exists("data/centre_A"))

def count_images(folder):
    total = 0
    for root, _, files in os.walk(folder):
        total += len([f for f in files if f.lower().endswith((".jpg",".jpeg",".png"))])
    return total

def analyse_centre(path):
    print(f"\nAnalysing: {path}")
    total_images = 0
    
    for class_name in os.listdir(path):
        class_path = os.path.join(path, class_name)
        
        if not os.path.isdir(class_path):
            continue
        
        print(f"\n  Class: {class_name}")
        
        if class_name == "positive": # Flatten positives
            count = count_images(class_path)
            print(f"    Total images: {count}")
            total_images += count
        
        elif class_name == "negative": # Sequence-based negatives
            seq_counts = []
            
            for seq in os.listdir(class_path):
                seq_path = os.path.join(class_path, seq)
                
                if not os.path.isdir(seq_path):
                    continue
                
                count = count_images(seq_path)
                seq_counts.append(count)
            
            if len(seq_counts) > 0:
                print(f"    Number of sequences: {len(seq_counts)}")
                print(f"    Average images per sequence: {sum(seq_counts)/len(seq_counts):.2f}")
                print(f"    Min images in sequence: {min(seq_counts)}")
                print(f"    Max images in sequence: {max(seq_counts)}")
                print(f"    Total negative images: {sum(seq_counts)}")
                
                total_images += sum(seq_counts)
        
    print(f"\n  Total images in centre: {total_images}\n")

# Run for all centres
analyse_centre("data/centre_A")
analyse_centre("data/centre_B")
analyse_centre("data/centre_C")