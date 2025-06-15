import numpy as np
import torch as th
import matplotlib.pyplot as plt
import csv

def one_hot(config, state) :
    """
    Cette méthode permet de coder une observation en One Hot Encoding.
    
    config : int ou list(int)
    Décrit l'espace d'observation du problème.
    
    state : int ou list(int)
    L'observation à coder
    
    Renvoie un array numpy avec l'observation codée selon le One Hot Encoding
    
    
    Exemple 1 :
    Soit un environnement d'espace d'observation : Discrete(3)
    
    Alors on définit
    config = 3
    
    L'appel one_hot(3, 0) renvoie : [1, 0, 0]
    L'appel one_hot(3, 2) renvoie : [0, 0, 1]
    
    
    Exemple 2 :
    Soit un environnemnt d'espace d'observation : MultiDiscrete(2, 3)
    
    Alors on définit
    config = [2, 3]
    
    L'appel one_hot([2, 3], [0, 1]) renvoie : [1, 0, 0, 1, 0]
    L'appel one_hot([2, 3], [0, 2]) renvoie : [1, 0, 0, 0, 1]
    L'appel one_hot([2, 3], [1, 0]) renvoie : [0, 1, 1, 0, 0]
    
    
    Attention à la définition de config qui doit rigoureusement suivre la définition de l'espace d'observation.
    
    Pour l'espace d'observation MultiDiscrete(2, 3) : config = [2, 3]
    Pour MultiDiscrete(3, 2) : config = [3, 2]
    """
    try :
        n = sum(config)
        obs = [0 for _ in range(n)]
        obs[int(state[0])] = 1
        k=0
        for i in range(1, len(config)) :
            k += config[i-1]
            obs[k+int(state[i])] = 1
    except TypeError :
        obs = [0 for _ in range(config)]
        obs[int(state)] = 1
    
    return np.array(obs, dtype = np.float32)



def render_one_episode(env, module, oneHotEncoding_config = None) :
    """
    Affiche un episode de l'environnement
    
    env : L'environnement
    
    module : Le réseau de l'agent qui intéragit avec l'environnement
    """
    episode_return = 0.0
    done = False
    obs, info = env.reset()
    
    while not done:
        env.render()

        # On code l'observation
        if oneHotEncoding_config is not None :
            obs_batch = th.from_numpy(one_hot(oneHotEncoding_config, obs)).unsqueeze(0)
        
        # Récupération de la sortie du réseau
        model_outputs = module.forward_inference({'obs': obs_batch})

        # Récupération de l'action à effectuer
        action_dist_params = model_outputs["actions"][0].numpy()
        
        greedy_action = action_dist_params

        # Application de l'action
        obs, reward, terminated, truncated, info = env.step(greedy_action)


        episode_return += reward
        done = terminated or truncated
    env.render()
    print(f"Récompense obtenue au cours de l'épisode : {episode_return}")
    return episode_return




def convert(valeur) :
    """
    Permet de convertir une chaîne de caractère issue
    du path en valeur de paramètre
    """
    separator_position = []
    composite = False
    for i in range(len(valeur)) :
        c = valeur[i]
        if c == "_" :
            composite = True
            separator_position.append(i)

    if composite :
        new_value = []
        old_position = -1
        for i in separator_position :
            new_value.append(int(valeur[old_position + 1 : i]))
            old_position = i
        new_value.append(int(valeur[old_position + 1 :]))
    else :
        new_value = float(valeur)
    return new_value




def get_combinaison(chemin) :
    """
    Permet d'obtenir un dictionnaire pour obtenir le paramétrage
    du modèle à partir du chemin de sauvegarde.
    """
    # On obtient la fin du chemin
    position_separateur = 0
    for i in range(len(chemin)) :
        c =  chemin[i]
        if c == '/' :
            position_separateur = i
    chemin = chemin[position_separateur + 1 :]

    # On retire les informations qui précèdent les paramètres
    position_separateur = 0
    compteur = 0
    for i in range(len(chemin)) :
        c = chemin[i]
        if c == '_' :
            compteur += 1
        if compteur == 5 :
            position_separateur = i
            break
    chemin = chemin[position_separateur + 1 :]


    # On individualise nos paramètres
    params = chemin.split(",")

    # On retire les informations qui suivent le dernier paramètre
    position_separateur = []
    for i in range(len(params[-1])) :
        c = params[-1][i]
        if c == "_" :
            position_separateur.append(i)
    params[-1] = params[-1][: position_separateur[-2]]
    

    # On sépare les valeurs des clefs
    for i in range(len(params)) :
        params[i] = params[i].split("=")

    # On convertit en dictionnaire
    params_dict = {key : convert(value) for (key, value) in params}
    return params_dict



def retrieve_rewards(path) :
    """
    Permet d'obtenir les récompenses au fil des steps à partir
    du chemin de sauvegarde d'un entraînement.
    """
    res = []
    maxi_rewards = []
    y = []
    with open(path + "/progress.csv") as csvFile :
        file = csv.DictReader(csvFile)
        maxi = None
        i = 0
        for row in file :
            i += int(row['num_training_step_calls_per_iteration'])
            if 'env_runners/episode_return_mean' in row :	# On vérifie qu'au moins une récompense soit enregistrée.
                y.append(i)
        
                mean_reward = float(row['env_runners/episode_return_mean'])
                res.append(mean_reward)
                if maxi is None :
                    maxi = mean_reward
        
                if maxi < mean_reward :
                    maxi = mean_reward
                maxi_rewards.append(maxi)
    return res, maxi_rewards, y


def comparison_plot_1(resultGrid, param1_id, param1_poss) :
    """
    Affichage des résultats pour la comparaison après GridSearch sur 2 paramètres.
    """

    n = len(param1_poss)

    fig, axes = plt.subplots(n, figsize=(10, 10))
    
    for result in resultGrid :
        path = result.path
        combinaison = get_combinaison(path)

        val1 = combinaison[param1_id]
        
        i = param1_poss.index(val1)

        rewards, maxi_rewards, y = retrieve_rewards(path)

        axes[i].plot(y, rewards, label='Récompense moyenne')
        axes[i].plot(y, maxi_rewards, label='Récompense moyenne maximale')
        axes[i].legend()
        axes[i].set_xlabel('Épisode')
        axes[i].set_ylabel('Récompense')
        axes[i].set_title(f"Récompense par épisode pour {param1_id} = {val1}")
        axes[i].grid()
    plt.show()




def comparison_plot_2(resultGrid, param1_id, param1_poss, param2_id, param2_poss) :
    """
    Affichage des résultats pour la comparaison après GridSearch sur 2 paramètres.
    """
    n = len(param1_poss)
    m = len(param2_poss)

    fig, axes = plt.subplots(m, n, figsize=(30, 30))
    
    for result in resultGrid :
        path = result.path
        combinaison = get_combinaison(path)

        val1 = combinaison[param1_id]
        val2 = combinaison[param2_id]
        
        i = param1_poss.index(val1)
        j = param2_poss.index(val2)

        rewards, maxi_rewards, y = retrieve_rewards(path)

        axes[j, i].plot(y, rewards, label='Récompense moyenne')
        axes[j, i].plot(y, maxi_rewards, label='Récompense moyenne maximale')
        axes[j, i].legend()
        axes[j, i].set_xlabel('Épisode')
        axes[j, i].set_ylabel('Récompense')
        axes[j, i].set_title(f"Récompense par épisode pour {param1_id} = {val1} et {param2_id} = {val2}")
        axes[j, i].grid()
    plt.show()
    
    
def comparison_plot_3(resultGrid, param1_id, param1_poss, param2_id, param2_poss, param3_id, param3_poss) :
    
    n = len(param1_poss)
    m = len(param2_poss)
    o = len(param3_poss)
    
    fig, axes = plt.subplots(m*o, n, figsize=(30, 30))
    
    for result in resultGrid :
        path = result.path
        combinaison = get_combinaison(path)

        val1 = combinaison[param1_id]
        val2 = combinaison[param2_id]
        val3 = combinaison[param3_id]
        
        i = param1_poss.index(val1)
        j = param2_poss.index(val2)
        k = param3_poss.index(val3)

        rewards, maxi_rewards, y = retrieve_rewards(path)

        axes[j+k*m, i].plot(y, rewards, label='Récompense moyenne')
        axes[j+k*m, i].plot(y, maxi_rewards, label='Récompense moyenne maximale')
        axes[j+k*m, i].legend()
        axes[j+k*m, i].set_xlabel('Épisode')
        axes[j+k*m, i].set_ylabel('Récompense')
        axes[j+k*m, i].set_title(f"Récompense par épisode pour {param1_id} = {val1}, {param2_id} = {val2} et {param3_id} = {val3}")
        axes[j+k*m, i].grid()
    plt.show()