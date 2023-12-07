 import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.losses import Huber
from tensorflow.keras.layers import Conv2D, Flatten
from tensorflow.keras.layers import Input, concatenate
from tensorflow.keras import Model
from collections import deque
import random
import pandas as pd
import re
class SumTree:
    """
    Estructura de datos SumTree para manejar el almacenamiento y muestreo de prioridades.
    """
    def __init__(self, capacity):
        self.capacity = capacity  # Capacidad máxima del SumTree
        self.tree = np.zeros(2 * capacity - 1)  # Árbol para almacenar prioridades
        self.data = np.zeros(capacity, dtype=object)  # Array para almacenar experiencias
        self.data_pointer = 0  # Puntero para saber dónde almacenar nuevas experiencias

    def add(self, priority, data):
        """
        Agrega una experiencia con su prioridad en el árbol y los datos.
        """
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data  # Inserta la experiencia en el array
        self.update(tree_idx, priority)  # Actualiza el árbol con la prioridad
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # Si supera la capacidad, reinicia el puntero
            self.data_pointer = 0

    def update(self, tree_idx, priority):
        """
        Actualiza el árbol con una nueva prioridad.
        """
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        while tree_idx != 0:  # Propaga el cambio a través del árbol
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        """
        Obtiene una experiencia con prioridad y su índice en el árbol.
        """
        parent_idx = 0
        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1
            if left_child_idx >= len(self.tree):  # Si llega al final del árbol
                leaf_idx = parent_idx
                break
            else:  # Búsqueda descendente en el árbol
                if v <= self.tree[left_child_idx]:
                    parent_idx = left_child_idx
                else:
                    v -= self.tree[left_child_idx]
                    parent_idx = right_child_idx
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    def total_priority(self):
        """
        Retorna la prioridad total almacenada en la raíz del árbol.
        """
        return self.tree[0]  # La raíz del árbol contiene la suma total de prioridades

class DoubleDQN:

    def __init__(self, state_space, action_space, max_memory_size=250000):
        self.state_space = state_space
        self.action_space = action_space
        self.gamma = 0.995
        self.epsilon = 0.995
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.learning_rate = 0.0018
        self.wins = 0
        self.losses = 0
        try:
            self.model = self.load_model('chicken_model_final3dosinput_huesos.keras')
        except:
            self.model = self.build_model()
        self.save_model('chicken_model_final3dosinput_huesos.keras') 
        self.target_model = self.build_model()
        self.update_target_model()
        self.pretrain_mode = False
        self.game_counter = 0
        self.past_positions = [] 
        self.real_bone_positions_df = pd.DataFrame(columns=[0, 1, 2])
        self.real_or_random = None 
        self.board= None
        self.memory = SumTree(max_memory_size)  # Inicializa SumTree

        
    def build_model(self):
        # Primera entrada: grilla del juego
        input_grid = Input(shape=(5, 5, 1), name='input_grid')
        x1 = Conv2D(32, (3, 3), activation='relu')(input_grid)
        x1 = Dropout(0.2)(x1)
        x1 = Conv2D(64, (3, 3), activation='relu')(x1)
        x1 = Dropout(0.2)(x1)
        x1 = Flatten()(x1)
        
        # Segunda entrada: características adicionales (wins, losses, bone_positions)
        input_features = Input(shape=(5,), name='input_features')  # Aquí hay 5 características adicionales
        x2 = Dense(32, activation='relu')(input_features)
        
        # Combinar las dos entradas
        combined = concatenate([x1, x2])
        
        # Capas densas finales
        x3 = Dense(128, activation='relu')(combined)
        output = Dense(self.action_space, activation='linear')(x3)
        
        model = Model(inputs=[input_grid, input_features], outputs=output)
        model.compile(loss=Huber(), optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        
        return model


    def update_target_model(self):
        tau = 0.008
        for i, model_weight in enumerate(self.model.get_weights()):
            self.target_model.get_weights()[i] = tau * model_weight + (1 - tau) * self.target_model.get_weights()[i]
        self.target_model.set_weights(self.target_model.get_weights())


    def act(self, current_state, open_positions, additional_features):
        if np.random.rand() <= self.epsilon:
            if not open_positions:  # Verificar si la lista está vacía
                # Manejar la situación, por ejemplo, terminar el juego o devolver una acción nula
                return None  # O manejarlo de otra manera adecuada
            return random.choice(open_positions)
        
        q_values = self.model.predict([current_state, additional_features])[0]
        filtered_q_values = np.full(self.action_space, -np.inf)
        
        for pos in open_positions:
            index = pos[0] * 5 + pos[1]  # Convertir la posición 2D en un índice 1D
            filtered_q_values[index] = q_values[index]
        
        best_action_index = np.argmax(filtered_q_values)
        best_action_coordinates = divmod(best_action_index, 5)  # Convertir de nuevo a coordenadas 2D
        
        return best_action_coordinates

    
    def remember(self, current_state, additional_features, action, reward, next_state, done):
        experience = (current_state, additional_features, action, reward, next_state, done)
        max_priority = np.max(self.memory.tree[-self.memory.capacity:])
        if max_priority == 0:
            max_priority = 1.0  # Prioridad inicial para nuevas experiencias
        self.memory.add(max_priority, experience)  # Agrega la experiencia con su prioridad al SumTree



    def replay(self, batch_size=64):
        if self.memory.data_pointer < batch_size:
            return

        minibatch_indices = []
        minibatch = []

        segment = self.memory.total_priority() / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            v = np.random.uniform(a, b)
            index, priority, data = self.memory.get_leaf(v)
            minibatch_indices.append(index)
            minibatch.append(data)

        for tree_index, (current_state, additional_features, action, reward, next_state, done) in zip(minibatch_indices, minibatch):
            target = reward
            if not done:
                action_next = np.argmax(self.model.predict([next_state, additional_features])[0])
                target = reward + self.gamma * self.target_model.predict([next_state, additional_features])[0][action_next]

            target_f = self.model.predict([current_state, additional_features])
            current_prediction = target_f[action[0], action[1]]
            target_f[action[0], action[1]] = target

            self.model.fit([current_state, additional_features], target_f, epochs=1, verbose=0)

            # Calculando el nuevo error TD
            new_priority = abs(target - current_prediction) + 0.01  # El valor 0.01 evita prioridades cero
            self.memory.update(tree_index, new_priority)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay



            
    def load_model(self, model_path):
        try:
            loaded_model = tf.keras.models.load_model(model_path)
            print("Modelo cargado exitosamente desde", model_path)
            return loaded_model
        except FileNotFoundError:
            print("Archivo del modelo no encontrado.")
        except Exception as e:
            print("Error al cargar el modelo:", str(e))



    def save_model(self, model_name):
        try:
            model_path = 'C:/Users/JT7/python programas/' + model_name
            self.model.save(model_path)
            print("Modelo guardado exitosamente en", model_path)
        except PermissionError:
            print("Error: No se tiene permiso para escribir en la ruta especificada.")
        except Exception as e:
            print("Error desconocido al guardar el modelo:", str(e))


    def generate_bone_positions(self):
        method = self.game_counter % 4
        if method == 0 or method == 2:
            self.bone_positions = [tuple(np.random.randint(0, 5, size=2)) for _ in range(3)]
        elif method == 1:
            row = np.random.randint(0, 5)
            cols = np.random.choice(range(5), 3, replace=False)
            self.bone_positions = [(row, col) for col in cols]
        elif method == 3:
            last_bone = self.past_positions[-1] if self.past_positions else None
            if last_bone:
                row, col = last_bone[0]
                self.bone_positions = [(row, (col + i) % 5) for i in range(1, 4)]
            else:
                self.bone_positions = [tuple(np.random.randint(0, 5, size=2)) for _ in range(3)]
        self.past_positions.append(self.bone_positions)
        self.game_counter += 1
        return self.bone_positions


    def start_game(self):
        self.bone_positions = self.generate_bone_positions()
        self.board = np.zeros((5, 5))
        for pos in self.bone_positions:
            self.board[pos] = -1
        self.open_positions = [(i, j) for i in range(5) for j in range(5) if self.board[i, j] == 0]
        print("Tablero inicial:")
        print(np.array2string(self.board, formatter={'int': lambda x: str(x).rjust(2)}))

        return self.open_positions



    def assign_rewards(self, action, result=None, consecutive_wins=0):
        if result == 'l':
            # Asignar una penalización fuerte por perder
            reward = -25
            # Reseteamos el conteo de victorias consecutivas
            consecutive_wins = 0
            return reward, consecutive_wins
        elif result == 'w':
            # Asignar una recompensa fuerte por ganar
            reward = 5
            # Incrementamos el conteo de victorias consecutivas
            consecutive_wins += 1
            return reward, consecutive_wins
        else:
            # Código original de asignación de recompensas
            reward = 0
            if any(np.array_equal(action, bone) for bone in self.bone_positions):
                reward = -25
            else:
                reward = 1
                if consecutive_wins >= 5:
                    reward += 1
                if not any(np.array_equal(action, past) for past in self.past_positions):
                    reward += 1
                min_distance = min([abs(action[0] - bone[0]) + abs(action[1] - bone[1]) for bone in self.bone_positions])
                distance_reward = min_distance
                reward += distance_reward
            return reward, consecutive_wins




    def play_game(self, real_or_random=None):
        # Inicializar variables
        result = None
        real_bone_positions = []
        last_played_position = None
        done_local = False
        consecutive_wins = 0
        action = None
        play_again = ""
        if real_or_random is None:
            real_or_random = self.real_or_random

        move_counter = 22
        update_frequency = 3
        move_since_last_update = 3

        # Inicializar el tablero
        open_positions = self.start_game()
        played_positions = set()

        while True:
            while not done_local:
                current_state = self.board.copy().reshape(1, self.board.shape[0], self.board.shape[1], 1)
                final_board = self.board.copy()
                for pos in played_positions:
                    final_board[pos] = 1
                if last_played_position is not None:
                    final_board[last_played_position] = 2

                additional_features = final_board.reshape(1, -1)

                # Comprobar si hay posiciones disponibles
                if not open_positions:
                    print("No quedan posiciones disponibles. El juego termina.")
                    done_local = True
                    break

                # Acción sugerida
                action = self.act(current_state, open_positions, additional_features)
                while action in played_positions:
                    open_positions.remove(action)
                    action = self.act(current_state, open_positions, additional_features)

                print(f"Suggested Move: {action}")
                last_played_position = action

                user_input = input("Accept suggested move (y/n)? ").lower()
                while user_input not in ["y", "n"]:
                    print("Invalid input. Please enter 'y' or 'n'.")
                    user_input = input("Accept suggested move (y/n)? ").lower()

                if user_input == "y":
                    next_state = self.board.copy().reshape(1, self.board.shape[0], self.board.shape[1], 1)
                    played_positions.add(action)
                    if action in open_positions:
                        open_positions.remove(action)
                    move_counter -= 1
                    move_since_last_update += 1

                    if move_since_last_update >= update_frequency:
                        self.update_target_model()
                        move_since_last_update = 0

                    result = input("Did you win or lose (w/l)? ").lower()
                    while result not in ["w", "l"]:
                        print("Invalid input. Please enter 'w' or 'l'.")
                        result = input("Did you win or lose (w/l)? ").lower()

                    if result == "w":
                        self.wins += 1
                        reward, consecutive_wins = self.assign_rewards(action, result, consecutive_wins)
                        print(f"Incrementing wins {self.wins}")
                        choice = input("Quieres retirar Ganancia o continuar (c/s)? ").lower()
                        while choice not in ["c", "s"]:
                            print("Invalid input. Please enter 'c' or 's'.")
                            choice = input("Quieres retirar Ganancia o continuar (c/s)? ").lower()

                        if choice == "c":
                            real_bone_positions = self.get_real_bone_positions()
                            self.real_bone_positions_df.loc[len(self.real_bone_positions_df)] = real_bone_positions
                            real_board = np.zeros_like(self.board)
                            for pos in real_bone_positions:
                                real_board[pos] = -1
                            print("Real board with real bone positions:")
                            print(np.array2string(real_board, formatter={'int': lambda x: str(x).rjust(2)}))

                            current_state = real_board.reshape(1, self.board.shape[0], self.board.shape[1], 1)
                            next_state = real_board.reshape(1, self.board.shape[0], self.board.shape[1], 1)

                            # Actualizar SumTree con el nuevo estado
                            self.remember(current_state, additional_features, action, reward, next_state, True)

                            self.replay()  # Asegúrate de que esta función también esté actualizada para usar SumTree
                            self.save_model('chicken_model_final3dosinput_huesos.keras')
                            self.wins = 0
                            self.bone_positions = []
                            open_positions = self.start_game()
                            played_positions = set()
                            move_counter = 22
                            done_local = True
                            play_again = input("Quieres seguir jugando (y/n)? ").lower()
                            while play_again not in ["y", "n"]:
                                print("Invalid input. Please enter 'y' or 'n'.")
                                play_again = input("Quieres seguir jugando (y/n)? ").lower()
                            if play_again == "n":
                                print("Regresando al menú principal.")
                                done_local = True
                            else:
                                done_local = False

                        if choice == "s":
                            if move_counter == 0:
                                print("Game Over! Has descubierto los 22 pollos!")
                                done_local = True
                                self.replay()  # Asegúrate de que esta función también esté actualizada para usar SumTree
                                self.save_model('chicken_model_final3dosinput_huesos.keras')
                                self.wins = 0
                            else:
                                done_local = False

                    elif result == "l":
                        self.losses += 1
                        reward, consecutive_wins = self.assign_rewards(action, result, consecutive_wins)
                        real_bone_positions = self.get_real_bone_positions()
                        self.real_bone_positions_df.loc[len(self.real_bone_positions_df)] = real_bone_positions
                        real_board = np.zeros_like(self.board)
                        for pos in real_bone_positions:
                            real_board[pos] = -1
                        print("Real board with real bone positions:")
                        print(np.array2string(real_board, formatter={'int': lambda x: str(x).rjust(2)}))

                        done_local = True
                        current_state = real_board.reshape(1, self.board.shape[0], self.board.shape[1], 1)
                        next_state = real_board.reshape(1, self.board.shape[0], self.board.shape[1], 1)

                        # Actualizar SumTree con el nuevo estado
                        self.remember(current_state, additional_features, action, reward, next_state, True)

                        self.replay()  # Asegúrate de que esta función también esté actualizada para usar SumTree
                        self.save_model('chicken_model_final3dosinput_huesos.keras')
                        self.wins = 0
                        self.bone_positions = self.generate_bone_positions()
                        open_positions = self.start_game()
                        played_positions = set()
                        self.consecutive_wins = 0
                        move_counter = 22
                        if done_local:
                            play_again = input("Quieres seguir jugando (y/n)? ").lower()
                            while play_again not in ["y", "n"]:
                                print("Invalid input. Please enter 'y' or 'n'.")
                                play_again = input("Quieres seguir jugando (y/n)? ").lower()

                            if play_again == "n":
                                print("Regresando al menú principal.")
                                return
                            else:
                                done_local = False
                                self.start_game()
            else:
                print("Regresando al menú principal.")
                return


    def simulated_game(self, real_or_random=None):
        real_bone_positions = []
        last_played_position = None
        done_local = False
        consecutive_wins = 0
        
        if real_or_random is None:
            real_or_random = self.real_or_random

        move_counter = 22
        update_frequency = 3
        move_since_last_update = 0

        open_positions = self.start_game()
        played_positions = set()

        while not done_local:
            current_state = self.board.copy().reshape(1, self.board.shape[0], self.board.shape[1], 1)
            additional_features = np.concatenate([
                np.array(open_positions).reshape(1, -1),
                np.array(real_bone_positions).reshape(1, -1),
                np.array([last_played_position if last_played_position else 0]).reshape(1, -1),
                np.array([self.wins]).reshape(1, -1),
                np.array([self.losses]).reshape(1, -1)
            ], axis=1)

            suggested_move = self.act(current_state, open_positions, additional_features)

            while suggested_move in played_positions:
                open_positions.remove(suggested_move)
                suggested_move = self.act(current_state, open_positions, additional_features)

            last_played_position = suggested_move
            played_positions.add(suggested_move)
            
            if suggested_move in open_positions:
                open_positions.remove(suggested_move)
                
            if any(np.array_equal(suggested_move, bone) for bone in self.bone_positions):
                self.losses += 1
                reward = -25  # o la penalización que decidas
                done = True
            else:
                reward = 1  # o la recompensa que decidas
                done = move_counter == 0

            next_state = self.board.copy().reshape(1, self.board.shape[0], self.board.shape[1], 1)
            self.remember(current_state, additional_features, suggested_move, reward, next_state, done)

            move_counter -= 1
            move_since_last_update += 1

            if move_since_last_update >= update_frequency:
                self.update_target_model()
                move_since_last_update = 0

            self.replay()

            if done:
                return reward > 0  # Retorna True si se ganó, False si se perdió

        return False  # En caso de terminar el bucle sin una condición explícita de ganar o perder

        
    def pretrain(self, episodes):
        self.pretrain_mode = True
        self.wins = 0  # Inicializamos las victorias y derrotas si no lo has hecho
        self.losses = 0
        episode_counter = 0  # Añadimos un contador de episodios
        move_since_last_update = 0  # Contador de movimientos desde la última actualización
        update_frequency = 3  # Frecuencia de actualización del modelo objetivo

        for _ in range(episodes):
            won = self.simulated_game(real_or_random="r")  # Pasa el valor de real_or_random aquí
            episode_counter += 1  # Incrementamos el contador de episodios
            print(f"Episodio {episode_counter} terminado.")  # Usamos el contador de episodios
            print(f"Juego ganado: {won}")
            print(f"Wins acumuladas: {self.wins}, Losses acumuladas: {self.losses}")
            
            self.replay()
            move_since_last_update += 1  # Incrementar contador después de cada movimiento

            if move_since_last_update >= update_frequency:  # Comprobar si es hora de actualizar
                self.update_target_model()  # Actualizar modelo objetivo
                move_since_last_update = 0  # Reiniciar contador

        print("Pre-entrenamiento completado...")
        print(f"Wins: {self.wins}, Losses: {self.losses}")  # Usar self.wins y self.losses aquí
        self.save_model('chicken_model_final3dosinput_huesos.keras')
        self.pretrain_mode = False


    def get_real_bone_positions(self):
        self.bone_positions = []
        for i in range(3):
            while True:
                row_col = input(f"Entra posición hueso {i + 1} : ")
                try:
                    row, col = map(int, row_col.split(","))
                    if 0 <= row < 5 and 0 <= col < 5:
                        new_position = (row, col)
                        if new_position not in self.bone_positions:
                            self.bone_positions.append(new_position)
                            break
                        else:
                            print("Posición ya seleccionada para otro hueso.")
                    else:
                        print("Posición fuera de rango. Debe estar entre 0 y 4.")
                except ValueError:
                    print("Entrada inválida. Usa el formato (x, y).")
        return self.bone_positions


    def main_menu(self):
        while True:
            print("===== Chicken Game =====")
            print("1. Juego Real")
            print("2. Pre-Entrenamiento")
            print("3. Ver Win/Loss Estadisticas")
            print("4. Salir")

            choice = ""
            while choice not in ['1', '2', '3', '4'] or choice == "":
                choice = input("Escoje tu Opcion entre 1, 2, 3, 4: ")
                if choice in ['1', '2', '3', '4'] and choice != "":
                    break
                else:
                    print("Opcion Invalida. Por favor solo 1, 2, 3, or 4.")

            self.load_model('chicken_model_final3dosinput_huesos.keras')

            if choice == '1':
                self.real_or_random = input("Deseas generar las posiciones de los huesos de forma real (r) o al azar (a)? ").lower()
                while self.real_or_random not in ["r", "a"]:
                    print("Entrada Inválida. Por favor, introduce 'r' o 'a'.")
                    self.real_or_random = input("Deseas generar las posiciones de los huesos de forma real (r) o al azar (a)? ").lower()

                self.play_game()
            elif choice == '2':
                self.real_or_random = input("Deseas generar las posiciones de los huesos de forma real (r) o al azar (a)? ").lower()
                while self.real_or_random not in ["r", "a"]:
                    print("Entrada Inválida. Por favor, introduce 'r' o 'a'.")
                    self.real_or_random = input("Deseas generar las posiciones de los huesos de forma real (r) o al azar (a)? ").lower()

                episodes = int(input("Entra el numero de Juegos de Preentrenamiento: "))
                self.pretrain(episodes)
            elif choice == '3':
                self.show_win_loss_statistics()
            elif choice == '4':
                break


    def show_win_loss_statistics(self):
        print(f"Wins: {self.wins}")
        print(f"Losses: {self.losses}")

if __name__ == "__main__":
    dqn = DoubleDQN(25, 25)
    dqn.main_menu()