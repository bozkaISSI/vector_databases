import sqlalchemy
from sqlalchemy import Integer, Float, String, Boolean, create_engine, select
from sqlalchemy.orm import Session, declarative_base, mapped_column, Mapped
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from typing import Optional
from pgvector.sqlalchemy import Vector

# Define the base class for the table definition
Base = declarative_base()

# Define the Games table structure
class Games(Base):
    __tablename__ = "games"
    __table_args__ = {'extend_existing': True}

    # The vector size produced by the model
    VECTOR_LENGTH = 512

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(256))
    description: Mapped[str] = mapped_column(String(4096))
    windows: Mapped[bool] = mapped_column(Boolean)
    linux: Mapped[bool] = mapped_column(Boolean)
    mac: Mapped[bool] = mapped_column(Boolean)
    price: Mapped[float] = mapped_column(Float)
    game_description_embedding: Mapped[list[float]] = mapped_column(Vector(VECTOR_LENGTH))


# Generate embeddings for game descriptions
checkpoint = "distiluse-base-multilingual-cased-v2"
model = SentenceTransformer(checkpoint)

def generate_embeddings(text: str) -> list[float]:
    return model.encode(text)

# Function to insert games into the database
def insert_games(engine, dataset, chunk_size=5000):
    descriptions = []
    metadata = []

    # Step 1: Prepare game metadata
    for game in tqdm(dataset, desc="Preparing game metadata"):
        game_description = game["About the game"] or ""
        name = game["Name"]
        windows = game["Windows"]
        linux = game["Linux"]
        mac = game["Mac"]
        price = game["Price"]

        if name and windows is not None and linux is not None and mac is not None and price is not None and game_description:
            descriptions.append(game_description)
            metadata.append((name, windows, linux, mac, price, game_description))

    # Step 2: Chunk processing
    for i in range(0, len(metadata), chunk_size):
        chunk_metadata = metadata[i:i + chunk_size]
        chunk_descriptions = [m[5] for m in chunk_metadata]

        # Generate embeddings for this chunk
        embeddings = model.encode(chunk_descriptions, batch_size=64, show_progress_bar=False)

        # Create Game objects
        game_records = []
        for j, (name, windows, linux, mac, price, description) in enumerate(chunk_metadata):
            embedding = embeddings[j]
            game_record = Games(
                name=name,
                description=description[:4096],  # truncate if needed
                windows=windows,
                linux=linux,
                mac=mac,
                price=price,
                game_description_embedding=embedding
            )
            game_records.append(game_record)

        # Step 3: Insert chunk into database
        with Session(engine) as session:
            session.bulk_save_objects(game_records)
            session.commit()

        print(f"✅ Inserted {min(i + chunk_size, len(metadata))}/{len(metadata)} games.")


# Function to find games similar to the given game
def find_game(
    engine: sqlalchemy.Engine,  # Correcting the sqlalchemy import here
    game_description: str,
    windows: Optional[bool] = None,
    linux: Optional[bool] = None,
    mac: Optional[bool] = None,
    price: Optional[int] = None
):
    with Session(engine) as session:
        game_embedding = generate_embeddings(game_description)

        query = (
            select(Games)
            .order_by(Games.game_description_embedding.cosine_distance(game_embedding))
        )

        if price:
            query = query.filter(Games.price <= price)
        if windows:
            query = query.filter(Games.windows == True)
        if linux:
            query = query.filter(Games.linux == True)
        if mac:
            query = query.filter(Games.mac == True)

        result = session.execute(query, execution_options={"prebuffer_rows": True})
        game = result.scalars().first()

        return game

# Load dataset and create the engine
from datasets import load_dataset

dataset = load_dataset("FronkonGames/steam-games-dataset")
columns_to_keep = ["Name", "Windows", "Linux", "Mac", "About the game", "Price"]
N = 40000
dataset = dataset["train"].select_columns(columns_to_keep).select(range(N))

# Set up the engine
db_url = "postgresql+psycopg://postgres:password@localhost:5555/similarity_search_service_db"
engine = create_engine(db_url)

# Ensure the table exists before inserting data
Base.metadata.create_all(engine)  # Create the table(s) if they don't exist yet

# Insert data into the database
insert_games(engine, dataset)

# Example: Search for a similar game with specific filtering criteria
game = find_game(engine, "This is a game about a hero who saves the world", price=10)
if game:
    print(f"Game: {game.name}")
    print(f"Description: {game.description}")

game = find_game(engine, game_description="Home decorating", price=20)
if game:
    print(f"Game: {game.name}")
    print(f"Description: {game.description}")

# Example with more strict filtering
game = find_game(engine, game_description="Home decorating", mac=True, price=5)
if game:
    print(f"Game: {game.name}")
    print(f"Description: {game.description}")
