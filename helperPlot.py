import matplotlib.pyplot as plt
from QLearningTrainingBuilder import QLearningTrainingBuilder
from DQN_TrainingBuilder import DQN_TrainingBuilder


def construct_all_possibilities1(param1_poss) :
    n = len(param1_poss)
    
    if n <= 0 :
        raise NeedAtLeastOnePossibilityException
    else :
        data = []
        for x in range(n) :
            data.append((param1_poss[x], x))
        return data


def construct_all_possibilities2(param1_poss, param2_poss) :
    n = len(param1_poss)
    m = len(param2_poss)
    
    if n <= 0 or m <= 0 :
        raise NeedAtLeastOnePossibilityException
    else :
        data = []
        for y in range(m) :
            for x in range(n) :
                data.append((param1_poss[x], param2_poss[y], x, y))
        return data


def construct_all_possibilities3(param1_poss, param2_poss, param3_poss) :
    n = len(param1_poss)
    m = len(param2_poss)
    k = len(param3_poss)
    
    if min(n, m, k) <= 0 :
        raise NeedAtLeastOnePossibilityException
    else :
        data = []
        if k > 0 :
            for z in range(k) :
                for y in range(m) :
                    for x in range(n) :
                        data.append((param1_poss[x], param2_poss[y], param3_poss[z], x, y, z))
        return data





def plot_q_learning_performance_forAll1(savePath, env, timesteps, plotDim, predefined, param1_id, param1_poss) :
    data = construct_all_possibilities1(param1_poss)
    n = len(param1_poss)
    
    fig, axes = plt.subplots(n, figsize=(15, 15))
    
    for param1, x in data :
        trainer = QLearningTrainingBuilder()
        trainer.set_parameters(predefined + [(param1_id, param1)])
        
        QTable, QrewardsTrain, QrewardsProd = trainer.train(env, timesteps)
        
        axes[x].plot(QrewardsTrain, label = 'Entraînement')
        axes[x].plot(QrewardsProd, label = 'Hors Entraînemnt')
        axes[x].legend()
        axes[x].set_xlabel("Épisode")
        axes[x].set_ylabel("Récompense")
        axes[x].set_title(f"Récompense par épisode pour {param1_id} = {param1}")
        axes[x].grid()
        axes[x].set_xlim([0, plotDim[0]])
        axes[x].set_ylim([plotDim[1], plotDim[2]])
    plt.savefig(savePath, bbox_inches='tight')
    plt.show()



def plot_q_learning_performance_forAll2(savePath, env, timesteps, plotDim, predefined, param1_id, param1_poss, param2_id, param2_poss) :
    data = construct_all_possibilities2(param1_poss, param2_poss)
    n = len(param1_poss)
    m = len(param2_poss)
    
    fig, axes = plt.subplots(m, n, figsize=(80, 80))
    
    for param1, param2, x, y in data :
        trainer = QLearningTrainingBuilder()
        trainer.set_parameters(predefined + [(param1_id, param1), (param2_id, param2)])
        
        QTable, QrewardsTrain, QrewardsProd = trainer.train(env, timesteps)
        
        axes[y, x].plot(QrewardsTrain, label = 'Entraînement')
        axes[y, x].plot(QrewardsProd, label = 'Hors Entraînemnt')
        axes[y, x].legend()
        axes[y, x].set_xlabel("Épisode")
        axes[y, x].set_ylabel("Récompense")
        axes[y, x].set_title(f"Récompense par épisode pour {param1_id} = {param1} et {param2_id} = {param2}")
        axes[y, x].grid()
        axes[y, x].set_xlim([0, plotDim[0]])
        axes[y, x].set_ylim([plotDim[1], plotDim[2]])
    plt.savefig(savePath, bbox_inches='tight')
    plt.show()


def plot_q_learning_performance_forAll3(savePath, env, timesteps, plotDim, predefined, param1_id, param1_poss, param2_id, param2_poss, param3_id, param3_poss) :
    data = construct_all_possibilities3(param1_poss, param2_poss, param3_poss)
    n = len(param1_poss)
    m = len(param2_poss)
    k = len(param3_poss)
    
    fig, axes = plt.subplots(m*k, n, figsize=(80, 80))
    
    for param1, param2, param3, x, y, z in data :
        trainer = QLearningTrainingBuilder()
        trainer.set_parameters(predefined + [(param1_id, param1), (param2_id, param2), (param3_id, param3)])
        
        QTable, QrewardsTrain, QrewardsProd = trainer.train(env, timesteps)
        
        axes[y+z*m, x].plot(QrewardsTrain, label = 'Entraînement')
        axes[y+z*m, x].plot(QrewardsProd, label = 'Hors Entraînemnt')
        axes[y+z*m, x].legend()
        axes[y+z*m, x].set_xlabel("Épisode")
        axes[y+z*m, x].set_ylabel("Récompense")
        axes[y+z*m, x].set_title(f"Récompense par épisode pour {param1_id} = {param1} et {param2_id} = {param2} et {param3_id} = {param3}")
        axes[y+z*m, x].grid()
        axes[y+z*m, x].set_xlim([0, plotDim[0]])
        axes[y+z*m, x].set_ylim([plotDim[1], plotDim[2]])
    plt.savefig(savePath, bbox_inches='tight')
    plt.show()



def plot_dqn_performance_forAll1(savePath, env, timesteps, plotDim, predefined, param1_id, param1_poss) :
    data = construct_all_possibilities1(param1_poss)
    n = len(param1_poss)
    
    fig, axes = plt.subplots(n, figsize=(15, 15))
    
    for param1, x in data :
        trainer = DQN_TrainingBuilder()
        trainer.set_parameters(predefined + [(param1_id, param1)])
        
        model, DQNrewardsTrain, DQNrewardsProd = trainer.train(env, timesteps)
        
        axes[x].plot(DQNrewardsTrain, label = 'Entraînement')
        axes[x].plot(DQNrewardsProd, label = 'Hors Entraînemnt')
        axes[x].legend()
        axes[x].set_xlabel("Épisode")
        axes[x].set_ylabel("Récompense")
        axes[x].set_title(f"Récompense par épisode pour {param1_id} = {param1}")
        axes[x].grid()
        axes[x].set_xlim([0, plotDim[0]])
        axes[x].set_ylim([plotDim[1], plotDim[2]])
    plt.savefig(savePath, bbox_inches='tight')
    plt.show()



def plot_dqn_performance_forAll2(savePath, env, timesteps, plotDim, predefined, param1_id, param1_poss, param2_id, param2_poss) :
    data = construct_all_possibilities2(param1_poss, param2_poss)
    n = len(param1_poss)
    m = len(param2_poss)
    
    fig, axes = plt.subplots(m, n, figsize=(80, 80))
    
    for param1, param2, x, y in data :
        trainer = DQN_TrainingBuilder()
        trainer.set_parameters(predefined + [(param1_id, param1), (param2_id, param2)])
        
        model, DQNrewardsTrain, DQNrewardsProd = trainer.train(env, timesteps)
        
        axes[y, x].plot(DQNrewardsTrain, label = 'Entraînement')
        axes[y, x].plot(DQNrewardsProd, label = 'Hors Entraînemnt')
        axes[y, x].legend()
        axes[y, x].set_xlabel("Épisode")
        axes[y, x].set_ylabel("Récompense")
        axes[y, x].set_title(f"Récompense par épisode pour {param1_id} = {param1} et {param2_id} = {param2}")
        axes[y, x].grid()
        axes[y, x].set_xlim([0, plotDim[0]])
        axes[y, x].set_ylim([plotDim[1], plotDim[2]])
    plt.savefig(savePath, bbox_inches='tight')
    plt.show()


def plot_dqn_performance_forAll3(savePath, env, timesteps, plotDim, predefined, param1_id, param1_poss, param2_id, param2_poss, param3_id, param3_poss) :
    data = construct_all_possibilities3(param1_poss, param2_poss, param3_poss)
    n = len(param1_poss)
    m = len(param2_poss)
    k = len(param3_poss)
    
    fig, axes = plt.subplots(m*k, n, figsize=(80, 80))
    
    for param1, param2, param3, x, y, z in data :
        trainer = DQN_TrainingBuilder()
        trainer.set_parameters(predefined + [(param1_id, param1), (param2_id, param2), (param3_id, param3)])
        
        model, DQNrewardsTrain, DQNrewardsProd = trainer.train(env, timesteps)
        
        axes[y+z*m, x].plot(DQNrewardsTrain, label = 'Entraînement')
        axes[y+z*m, x].plot(DQNrewardsProd, label = 'Hors Entraînemnt')
        axes[y+z*m, x].legend()
        axes[y+z*m, x].set_xlabel("Épisode")
        axes[y+z*m, x].set_ylabel("Récompense")
        axes[y+z*m, x].set_title(f"Récompense par épisode pour {param1_id} = {param1} et {param2_id} = {param2} et {param3_id} = {param3}")
        axes[y+z*m, x].grid()
        axes[y+z*m, x].set_xlim([0, plotDim[0]])
        axes[y+z*m, x].set_ylim([plotDim[1], plotDim[2]])
    plt.savefig(savePath, bbox_inches='tight')
    plt.show()
