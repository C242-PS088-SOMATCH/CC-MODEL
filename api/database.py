import mysql.connector
from mysql.connector import Error

def get_image_data_from_db(catalog_id):
    try:
        connection = mysql.connector.connect(
            host='your_database_host',
            database='your_database_name',
            user='your_database_user',
            password='your_database_password'
        )
        
        cursor = connection.cursor(dictionary=True)
        query = "SELECT * FROM mycatalog WHERE id = %s"
        cursor.execute(query, (catalog_id,))
        result = cursor.fetchone()
        
        return result if result else None
    except Error as e:
        print(f"Error fetching data from MySQL: {e}")
        return None
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
