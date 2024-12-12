import mysql.connector
from mysql.connector import Error
import os

def create_connection():
    """Create a new database connection."""
    return mysql.connector.connect(
        host=os.getenv('DB_HOST'),
        database=os.getenv('DB_NAME'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD')
    )

def get_image_data_from_db(catalog_id):
    """Fetch a specific image data from the database using the provided connection."""
    try:
        connection = create_connection()
        cursor = connection.cursor(dictionary=True)
        query = "SELECT * FROM mycatalog WHERE id = %s"
        cursor.execute(query, (catalog_id,))
        result = cursor.fetchone()
        return result if result else None
    except Error as e:
        print(f"Error fetching data from MySQL: {e}")
        return None
    finally:
        cursor.close()

def get_all_image_data_from_db():
    """Fetch all image data from the database using the provided connection."""
    try:
        connection = create_connection()
        cursor = connection.cursor(dictionary=True)
        query = "SELECT * FROM mycatalog"
        cursor.execute(query)
        result = cursor.fetchall()
        return result if result else None
    except Error as e:
        print(f"Error fetching data from MySQL: {e}")
        return None
    finally:
        cursor.close()
        
def set_feature_image(dominant_color, style, color_grup, name, value):
    """Fetch all image data from the database using the provided connection."""
    try:
        connection = create_connection() 
        cursor = connection.cursor(dictionary=True)
        query = "UPDATE mycatalog SET dominant_color = %s, style = %s, color_group = %s, name = %s WHERE id = %s"
        cursor.execute(query, (dominant_color, style, color_group, name, value))
        result = cursor.fetchall()
        return result if result else None
    except Error as e:
        print(f"Error fetching data from MySQL: {e}")
        return None
    finally:
        cursor.close()
