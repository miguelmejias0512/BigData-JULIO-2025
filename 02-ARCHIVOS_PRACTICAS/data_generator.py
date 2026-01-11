import sqlite3
import time
import datetime
import random

DB_NAME = 'company_sensors.db'

def create_connection():
    """Crea una conexión a la base de datos SQLite y habilita WAL mode."""
    conn = None
    try:
        conn = sqlite3.connect(DB_NAME)
        conn.execute('PRAGMA journal_mode=WAL;')
        # print("WAL mode enabled for data generator.")
    except sqlite3.Error as e:
        print(f"Error al conectar a la base de datos o al habilitar WAL: {e}")
    return conn

def create_table(conn):
    """Crea la tabla sensor_data si no existe."""
    try:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sensor_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                sensor_id TEXT NOT NULL,
                sensor_type TEXT NOT NULL,
                reading_float REAL,
                reading_text TEXT,
                reading_int INTEGER
            );
        """)
        conn.commit()
    except sqlite3.Error as e:
        print(f"Error al crear la tabla: {e}")

def generate_humidity_data(sensor_id="HUMID_CTRL_001"):
    """Genera una lectura de humedad simulada."""
    humidity = round(random.uniform(30.0, 90.0), 1)
    return datetime.datetime.now(datetime.timezone.utc).isoformat(), sensor_id, "humidity", humidity, None, None

def generate_fire_data(sensor_id="FIRE_DET_001"):
    """Genera una lectura de detección de incendios simulada."""
    possible_status = ["Normal", "Alerta Leve", "Fuego Detectado"]
    status = random.choices(possible_status, weights=[0.95, 0.04, 0.01], k=1)[0] # Tomamos el primer elemento
    return datetime.datetime.now(datetime.timezone.utc).isoformat(), sensor_id, "fire_detection", None, status, None

def generate_motion_data(sensor_id="MOTION_OP_001"):
    """Genera una lectura de movimiento de operador simulada (conteo de eventos)."""
    events = random.randint(0, 10)
    return datetime.datetime.now(datetime.timezone.utc).isoformat(), sensor_id, "operator_movement", None, None, events

def insert_data(conn, data_tuple):
    """Inserta una fila de datos en la tabla sensor_data."""
    sql = '''INSERT INTO sensor_data(timestamp, sensor_id, sensor_type, reading_float, reading_text, reading_int)
             VALUES(?,?,?,?,?,?)'''
    try:
        cursor = conn.cursor()
        cursor.execute(sql, data_tuple)
        conn.commit()
    except sqlite3.Error as e:
        print(f"Error al insertar datos: {e}")

def main():
    conn = create_connection()
    if conn is not None:
        create_table(conn)
        
        # Lista de simuladores de sensores y sus argumentos
        sensor_simulators = [
            (generate_humidity_data, {"sensor_id": "HUMID_CTRL_001"}),
            (generate_humidity_data, {"sensor_id": "HUMID_CTRL_002"}), # Otro controlador de humedad
            (generate_fire_data, {"sensor_id": "FIRE_DET_001"}),
            (generate_fire_data, {"sensor_id": "FIRE_DET_HALL_A"}),   # Otro detector de incendio
            (generate_motion_data, {"sensor_id": "MOTION_OP_001"}),
            (generate_motion_data, {"sensor_id": "MOTION_SENSOR_ENTRANCE"}), # Otro sensor de movimiento
        ]
        
        print("Iniciando generación de datos de sensores...")
        try:
            while True:
                simulator_func, kwargs = random.choice(sensor_simulators)
                data_to_insert = simulator_func(**kwargs)
                
                insert_data(conn, data_to_insert)
                
                # Descomentar para ver los datos insertados en la consola
                # print(f"Insertado: TS='{data_to_insert[0]}', Sensor='{data_to_insert[1]}', Tipo='{data_to_insert[2]}', Float={data_to_insert[3]}, Text='{data_to_insert[4]}', Int={data_to_insert[5]}")
                
                time.sleep(random.uniform(0.5, 2.0)) # Intervalo de generación de datos reducido para más variedad
        except KeyboardInterrupt:
            print("\nGeneración de datos detenida por el usuario.")
        finally:
            if conn:
                conn.close()
                print("Conexión a SQLite cerrada en data_generator.")
    else:
        print("No se pudo crear la conexión a la base de datos en data_generator.")

if __name__ == '__main__':
    main()