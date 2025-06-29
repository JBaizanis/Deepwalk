import argparse

import networkx as nx
import random
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from torch.utils.data import Dataset, DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist, squareform

# Ορίζω ως seed το 42 στις γεννήτριες ψευδοτυχαίων αριθμών, ώστε τα αποτελέσματα να είναι επαναλήψιμα και να μην είναι διαφορετικά σε κάθε επανάληψη εκτέλεσης.
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 1. Παραγωγή random walks.
def generate_random_walks(graph, num_walks, walk_length):
    walks = [] # Αυτή η λίστα θα περιέχει τα random walks που θα παραχθούν.
    for node in graph.nodes: # Για κάθε κόμβο του γράφου θα δημιουργηθούν num_walks, τυχαία μονοπάτια που ξεκινούν από κάθε κόμβο.
        for _ in range(num_walks):
            walk = [node]
            current_node = node # Ο current_node αποθηκεύει τον τρέχοντα κόμβο που εξετάζεται στο μονοπάτι.
            for _ in range(walk_length -1):
                neighbors = list(graph.neighbors(current_node)) # Η λίστα των γειτόνων του current_node. Αν ο κόμβος δεν έχει γείτονες το μονοπάτι σταματά.
                if not neighbors:
                    break
                current_node = random.choice(neighbors) # Επιλέγεται τυχαία ένας από τους γείτονες του κόμβου και αυτός ο κόμβος προστίθεται στο μονοπάτι.
                walk.append(current_node)
            walks.append(walk) # Το ολοκληρωμένο random walk προστίθεται στη λίστα walks.
    return walks # Επιστρέφω τα random walks που δημιουργήθηκαν.

# 2. CBOW Dataset.
class CBOWDataset(Dataset):
    def __init__(self, walks, window_size, vocab_size): # Η κλάση CBOWDataset, κληρονομεί από την κλάση Dataset της PyTorch και υλοποιεί δύο βασικές μεθόδους,
                                                        # την _len_ και την _getitem_. Η _init_ (μέθοδος αρχικοποίησης) δέχεται την λίστα με τα random walks
                                                        # που παράχθηκαν προηγουμένως, το μέγεθος του παραθύρου για το CBOW, και το μέγεθος του λεξιλογίου
                                                        # (πλήθος μοναδικών κόμβων στον γράφο).
        self.data = [] # Λίστα που θα περιέχει όλα τα ζεύγη (context, target) για το CBOW.
        self.vocab_size = vocab_size
        half_window = (window_size - 1) // 2 # Το μισό μέγεθος του παραθύρου, δηλαδή πόσοι κόμβοι θα ληφθούν πριν και μετά τον target κόμβο,
                                             # όπως ορίζεται και στην εκφώνηση της εργασίας.
        for walk in walks:
            for i in range(half_window, len(walk) - half_window): # Για κάθε walk στη λίστα, παίρνω το παράθυρο από τη θέση (i-half_window) μέχρι και τη θέση
                                                                  # (i+half_window), παραλείποντας τον κόμβο i.
                context = walk[i - half_window:i] + walk[i + 1: i + half_window + 1] # Οι κόμβοι που βρίσκονται πριν και μετά τον κόμβο στόχο (target) μέσα στο
                                                                                     # παράθυρο.
                target = walk[i] # Ο κόμβος στο κέντρο του παραθύρου.
                self.data.append((context, target)) # Το ζεύγος (context, target) προστίθεται στη λίστα self.data. Π.χ για ένα walk [a,b,c,d,e] και window_size = 3
                                                    # παράγονται: context=[a,c], target=[b], context=[b,d], target=[c], context=[c,e], target=[d].

    def __len__(self):
        return len(self.data) # Επιστρέφει το πλήθος των δειγμάτων στο dataset, δηλαδή το πλήθος των (context, target) ζευγών.

    def __getitem__(self, idx): # Παίρνει το δείγμα που βρίσκεται στη θέση idx από τη λίστα self.data και μετατρέπει το context και το target σε tensors τύπου long.
                                # Επιστρέφει τα δύο tensors που θα χρησιμοποιηθούν για εκπαίδευση με την PyTorch. Αυτή η μετατροπή γίνεται ώστε τα δεδομένα να είναι
                                # συμβατά με τις λειτουργίες του PyTorch και να αναγνωρίζονται από τα layers του PyTorch όπως το embedding layer, το οποίο χρειάζεται
                                # indice σε μορφή long.
        context, target = self.data[idx]
        context_tensor = torch.tensor(context, dtype=torch.long)
        target_tensor = torch.tensor(target, dtype=torch.long)
        return context_tensor, target_tensor

# 3. CBOW Model.
class CBOWModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOWModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim) # Δημιουργία ενός embedding layer, που μετατρέπει έναν ακέραιο (index ενός κόμβου) σε μια
                                                                  # συνεχή διανυσματική αναπαράσταση (embedding), διαστάσεων embedding_dim. Ουσιαστικά, κάθε κόμβος
                                                                  # έχει το δικό του embedding vector.
        self.linear = nn.Linear(embedding_dim, vocab_size) # Δημιουργία ενός πλήρους συνδεδεμένου layer. Η είσοδος είναι ο μέσος των embedding vectors και η έξοδος
                                                           # ένα vector μήκους vocab_size που αντιπροσωπεύει πιθανότητες για κάθε λέξη/κόμβο στο λεξιλόγιο.
        self.softmax = nn.Softmax(dim=1) # Εφαρμογή της συνάρτησης Softmax, στη διάσταση 1 (δεύτερη διάσταση του tensor). Μετατροπή των τιμών από linear σε
                                         # πιθανότητες.

    def forward(self, context):
        embedded = self.embeddings(context)
        hidden = embedded.mean(dim=1) # Υπολογισμός του μέσου όρου κατά μήκος της διάστασης των context κόμβων.
        output = self.linear(hidden) # Το hidden vector (μέσος όρος των embedding vectors) περνάει από το πλήρως συνδεδεμένο layer.
        return self.softmax(output) # Το output περνάει από τη συνάρτηση Softmax για να μετατραπεί σε πιθανότητες. Η έξοδος είναι ένα tensor διαστάσεων
                                    # (batch_size, vocab_size) όπου κάθε γραμμή είναι πιθανότητες για κάθε κόμβο στο λεξιλόγιο.

    def get_embeddings(self):
        return self.embeddings.weight.data # Επιστρέφει τα weights του embedding layer, δηλαδή τα embeddings για όλους τους κόμβους.

# 4. Συνάρτηση Εκπαίδευσης
def train_model(model, dataloader, epochs, learning_rate):
    # model: το μοντέλο προς εκπαίδευση, dataloader: batches δεδομένων, δηλαδή context και target ζευγάρια για την εκπαίδευση, epochs: Ο αριθμός των φορών που
    # το dataset θα περαστεί από το μοντέλο κατά τη διάρκεια της εκπαίδευσης.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs): # Ένα epoch (εποχή) είναι ένα πλήρες πέρασμα του dataset. Θα κάνω λοιπόν epochs περάσματα.
        total_loss = 0 # Θα μετρήσω το loss σε κάθε epoch για να δω την απόδοση κατά την διάρκεια της εκπαίδευσης.

        for context, target in dataloader: # Κάνω iterate τα batches δεδομένων που περιέχονται στον dataloader. Για κάθε batch παίρνω το context και το target.
            optimizer.zero_grad()
            output = model(context) # Το context tensor, παιρνιέται στο μοντέλο (CBOW model). Αυτό κάνει trigger την forward μέθοδο στην κλάση CBOWModel, η οποία
                                    # υπολογίζει τις πιθανότητες για κάθε λέξη/κόμβο στο λεξικό.
            loss = criterion(output, target) # Το output (οι πιθανότητες) και η "λέξη στόχος (κόμβος)" που θέλουμε να βρούμε παιρνιούνται στο loss function
                                             # nn.CrossEntropyLoss(). Αυτή υπολογίζει την απώλεια (loss) ή error ανάμεσα στις προβλέψεις του μοντέλου και του
                                             # πραγματικού target.
            loss.backward() # Η συνάρτηση backward(), υπολογίζει τα gradients του loss, σε συνάρτηση με τις παραμέτρους του μοντέλου χρησιμοποιώντας backpropagation.

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            total_loss += loss.item() # Το loss για αυτό το batch προστίθεται για αυτό το epoch στην total_loss.

        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}") # Τυπώνω στο τέλος κάθε epoch, το total_loss. Αυτό δείχνει πόσο καλά το μοντέλο μαθαίνει. Μια πτώση
                                                            # ανάμεσα στα epochs ορίζει ότι το μοντέλο βελτιώνεται.

# 5. Οπτικοποίηση με t-SNE
def visualize_embeddings(embeddings, graph):
    tsne = TSNE(n_components=2, random_state=42) # Η t-SNE τεχνική χρησιμοποιείται για dimensionality reduction. Κρατά τις μεταξύ αποστάσεις μεταξύ των σημείων
                                                 # στον κανονικό χώρο, σε έναν νέο μικρότερων διαστάσεων, εδώ 2D.
    reduced_embeddings = tsne.fit_transform(embeddings)
    plt.figure(figsize=(10, 10))

    # Τυπώνω τους κόμβους σε ένα plot, όπου κάθε σημείο θα αναπαριστά ένα κόμβο του γραφήματος. Θα εμφανίζονται δηλαδή οι σχετικές θέσεις των κόμβων,
    # με βάση τα embeddings τους.
    for i, label in enumerate(graph.nodes):
        x, y = reduced_embeddings[i, :]
        plt.scatter(x, y)
        plt.annotate(str(label), (x, y), textcoords='offset points', xytext=(5, 2), ha='center')

    plt.title("t-SNE Visualization of Node Embeddings")
    plt.show()

def calculate_distances(graph, embeddings):
    # Υπολογισμός των αποστάσεων των συντομότερων διαδρομών μεταξύ όλων των ζευγών κόμβων στο γράφημα Zachary Karate Club.
    shortest_path_distances = dict(nx.all_pairs_shortest_path_length(graph))
    # Υπολογισμός των ευκλείδειων αποστάσεων μεταξύ όλων των ζευγών των embeddings.
    embedding_distances = squareform(pdist(embeddings, metric='euclidean'))

    # Δημιουργία λίστας με ζεύγη αποστάσεων για το γράφημα και τα embeddings
    graph_distances = []
    embedding_distances_list = []

    # Επανάληψη για κάθε ζεύγος κόμβων (i, j) στο γράφημα Zachary Karate Club.
    for i in range(len(graph.nodes)):
        for j in range(i + 1, len(graph.nodes)):
            # Προσθήκη της απόστασης συντομότερης διαδρομής στο γράφημα Zachary Karate Club στη λίστα graph_distances.
            graph_distances.append(shortest_path_distances[i][j])
            # Προσθήκη της ευκλείδειας απόστασης των embeddings στη λίστα embedding_distances_list.
            embedding_distances_list.append(embedding_distances[i][j])

    # Υπολογισμός και εκτύπωση του συντελεστή συσχέτισης Pearson μεταξύ των δύο λιστών αποστάσεων.
    correlation, _ = pearsonr(graph_distances, embedding_distances_list)
    print(f"Pearson Correlation Coefficient: {correlation:.4f}")

def kMeans(embeddings, n_clusters, graph):
    # Κάνω kMeans Clustering ώστε να δω τα embeddings των κόμβων του γραφήματος Zachary Karate Club, αν θα καταλήξουν στο ίδιο cluster (ομάδα) όπως στο Zachary Karate Club.
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, init='k-means++')
    labels = kmeans.fit_predict(embeddings)

    # Βρίσκω τους κόμβους στο Zachary Karate Club που έχουν label 0 και 1 αντίστοιχα ώστε να τους εντάξω στις αντίστοιχες κοινότητες (community_0 -> Officer και community_1 -> Mr.Hi)
    community_0 = [node for i, node in enumerate(graph.nodes) if labels[i] == 0]
    community_1 = [node for i, node in enumerate(graph.nodes) if labels[i] == 1]

    # Αφού έχω εκτελέσει τον kmeans για τα embeddings στις αρχικές τους διαστάσεις, για λόγους οπτικοποίησης μειώνω τις διαστάσεις σε 2.
    tsne = TSNE(n_components=2, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)

    # Για κάθε community (officer και mr.hi) παίρνω τα embeddings. Πιο συγκεκριμένα για κάθε embedding κόμβου, βλέπω ποια ανήκουν στην ομάδα 0 και τα αποθηκεύω στην μεταβλητή
    # officer_embeddings. Αντίστοιχα βλέπω ποια ανήκουν στην ομάδα 1 και τα αποθηκεύω στην μεταβλητή MrHi_embeddings.
    officer_embeddings = reduced_embeddings[labels == 0]
    MrHi_embeddings = reduced_embeddings[labels == 1]

    # Κάνω plot τα Officer Embeddings
    plt.scatter(officer_embeddings[:, 0], officer_embeddings[:, 1], c='blue', label='Officer Embeddings')

    # Κάνω plot τα Mr.Hi Embeddings
    plt.scatter(MrHi_embeddings[:, 0], MrHi_embeddings[:, 1], c='red', label='Mr.Hi Embeddings')

    # Για κάθε σημείο (Embedding) αναθέτω την τιμή του κόμβου όπως είναι στο γράφημα Zachary Karate Club, για να είναι πιο κατανοητή η οπτικοποίηση.
    for i, node in enumerate(community_0):
        plt.text(officer_embeddings[i, 0], officer_embeddings[i, 1], str(node), fontsize=9, color='blue')

    for i, node in enumerate(community_1):
        plt.text(MrHi_embeddings[i, 0], MrHi_embeddings[i, 1], str(node), fontsize=9, color='red')

    plt.title("k-Means Clustering on Word Embeddings (Zachary Karate Club)")
    plt.legend()
    plt.show()

def main():
    # Παράμετροι χρήστη από το terminal για την λειτουργία του προγράμματος.
    parser = argparse.ArgumentParser(description="Generate random walks on Zachary Karate Club undirected graph")
    parser.add_argument("num_walks", type=int, help="Number of random walks per node")
    parser.add_argument("walk_length", type=int, help="Length of each random walk")

    parser.add_argument("window_size", type=int, help="Window size for CBOW")
    parser.add_argument("embedding_dim", type=int, help="Dimension of embeddings")
    parser.add_argument("batch_size", type=int, help="Number of batches per epoch")
    parser.add_argument("epochs", type=int, help="Number of training epochs")
    parser.add_argument("learning_rate", type=float, help="Learning rate")
    args = parser.parse_args()
    # Ελέγχω τις βασικές τιμές των παραμέτρων που δίνει ο χρήστης στην κονσόλα.
    if (args.window_size < 3):
        print("Window size must be greater than or equal to 3!")
        exit(0)
    if (args.embedding_dim < 10 or args.embedding_dim > 16 ):
        print("Embedding dimensionality should be must be between 10 and 16!")
        exit(0)
    if (args.learning_rate > 0.01):
        print("Learning rate should be lower or equal to 0.01!")
        exit(0)

    # Φορτώνω το Zachary Karate Club γράφημα.
    graph = nx.karate_club_graph()
    # Εκτυπώνω ποιοι κόμβοι συνδέονται με ποιους κόμβους στο Zachary Karate Club γράφημα για λόγους debugging.
    for edge in graph.edges:
        print(edge)

    # Δημιουργία τυχαίων περιπάτων από τους κόμβους του παραπάνω γραφήματος.
    num_walks = args.num_walks
    walk_length = args.walk_length
    random_walks = generate_random_walks(graph, num_walks, walk_length)

    # Προετοιμασία του CBOW dataset.
    window_size = args.window_size
    vocab_size = len(graph.nodes)
    embedding_dim = args.embedding_dim
    dataset = CBOWDataset(random_walks, window_size, vocab_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Αρχικοποίηση του CBOW μοντέλου.
    model = CBOWModel(vocab_size, embedding_dim)

    # Εκπαίδευση του μοντέλου.
    epochs = args.epochs
    learning_rate = args.learning_rate
    train_model(model, dataloader, epochs, learning_rate)

    # Κάνω ένα KMeans clustering ώστε να δω εν τέλει αν τα embeddings των κόμβων του Zachary Karate Club "πέφτουν" στο ίδιο cluster όπως στο Zachary Karate Club, για να δω
    # την επιτυχία του μοντέλου.
    embeddings = model.get_embeddings()
    kMeans(embeddings, 2, graph)
    # Υπολογισμός συσχέτισης μεταξύ των αποστάσεων του γραφήματος Zachary Karate Club και αποστάσεων embeddings.
    calculate_distances(graph, embeddings)
    # Οπτικοποίηση Embeddings στον 2D χώρο χωρίς την χρήση k-Means.
    visualize_embeddings(embeddings, graph)

if __name__ == "__main__":
    main()