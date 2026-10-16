#!/usr/bin/env python3
"""
app.py — Interface Web Interactive Streamlit pour Observational Reinforcement Learning.

Banc d'essai expérimental pour le sujet de recherche :
« Apprentissage par renforcement observationnel : théorie, architectures et dynamique d'apprentissage social »

Pages :
- Page A — Visualisation GridWorld (animation pas-à-pas, commandes interactives)
- Page B — Entraînement (configuration, suivi dynamique en temps réel)
- Page C — Comparaison (benchmark simultané des trois stratégies d'apprentissage)
- Page D — Résultats & Figures (exploration des CSV et galerie des graphiques)
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional

# Assurer l'encodage UTF-8 et la découverte des modules depuis la racine
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from observational_rl.environments.gridworld import GridWorldEnv
from observational_rl.training.trainer import evaluate_agent
from observational_rl.ui.helpers import (
    ACTION_NAMES_FR,
    PRESET_ENVS,
    create_ui_agent,
    create_ui_environment,
    list_available_figures,
    load_csv_results,
    render_grid_figure,
    step_simulation,
    train_ui_agent,
)

# Configuration globale Streamlit
st.set_page_config(
    page_title="Observational RL — Banc d'Essai",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Répertoires de données
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
CONFIGS_DIR = PROJECT_ROOT / "configs"


# ─────────────────────────────────────────────────────────────────────────────
# Initialisation de l'état de session (st.session_state)
# ─────────────────────────────────────────────────────────────────────────────
def init_session_state() -> None:
    if "env" not in st.session_state:
        st.session_state.env = create_ui_environment("Facile (5x5, 2 obstacles)")
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "agent_name" not in st.session_state:
        st.session_state.agent_name = "Q-Learning"
    if "trajectory" not in st.session_state:
        st.session_state.trajectory = [st.session_state.env.agent_pos]
    if "total_reward" not in st.session_state:
        st.session_state.total_reward = 0.0
    if "step_count" not in st.session_state:
        st.session_state.step_count = 0
    if "done" not in st.session_state:
        st.session_state.done = False
    if "last_action" not in st.session_state:
        st.session_state.last_action = "Aucune (position initiale)"
    if "status_message" not in st.session_state:
        st.session_state.status_message = "Prêt. Cliquez sur 'Étape suivante' ou 'Démarrer'."
    if "trained_agents_cache" not in st.session_state:
        st.session_state.trained_agents_cache = {}


init_session_state()


# ─────────────────────────────────────────────────────────────────────────────
# Barre latérale : Navigation & Contexte scientifique
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.title("🧭 Observational RL")
st.sidebar.markdown(
    "**Thèse de Doctorat** : *Apprentissage par renforcement observationnel : "
    "théorie, architectures et dynamique d'apprentissage social*"
)
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "🧭 Page A — Visualisation GridWorld",
        "⚡ Page B — Entraînement",
        "📊 Page C — Comparaison",
        "📁 Page D — Résultats & Figures",
    ],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.info(
    "💡 **Paradigmes comparés** :\n"
    "- **Pure RL** : Q-Learning tabulaire (exploration autonome)\n"
    "- **Pure Imitation** : Behavioral Cloning (supervisé sur démonstrations)\n"
    "- **Hybrid ORL** : Warm-start logit BC + ajustement online TD"
)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE A : Visualisation GridWorld pas-à-pas
# ─────────────────────────────────────────────────────────────────────────────
if page == "🧭 Page A — Visualisation GridWorld":
    st.header("🧭 Page A — Visualisation GridWorld Interactive")
    st.markdown(
        "Observez le comportement de navigation de l'agent pas-à-pas. "
        "L'environnement utilise les véritables transitions de `GridWorldEnv` et les politiques d'actions réelles."
    )

    col_cfg, col_view = st.columns([1, 2])

    with col_cfg:
        st.subheader("⚙️ Configuration")

        preset_choice = st.selectbox(
            "Environnement",
            list(PRESET_ENVS.keys()) + ["Personnalisé"],
            index=0,
            key="page_a_preset",
        )

        if preset_choice == "Personnalisé":
            grid_size = st.slider("Taille de la grille (N x N)", 4, 10, 6, key="page_a_gs")
            n_obs = st.slider("Nombre d'obstacles", 0, grid_size * grid_size - 4, 3, key="page_a_nobs")
            obs_seed = st.number_input("Graine obstacles", value=42, step=1, key="page_a_obs_seed")
            env_inst = create_ui_environment(
                preset_name="Personnalisé",
                custom_size=grid_size,
                custom_obstacles=n_obs,
                obstacle_seed=int(obs_seed),
            )
        else:
            env_inst = create_ui_environment(preset_name=preset_choice)

        agent_type = st.selectbox(
            "Type d'agent",
            ["Q-Learning", "Behavioral Cloning", "Hybrid ORL"],
            index=0,
            key="page_a_agent_type",
        )

        agent_source = st.radio(
            "Origine du modèle",
            ["Agent entraîné (démonstration rapide)", "Agent non entraîné (aléatoire)"],
            index=0,
            key="page_a_agent_source",
        )

        # Cache key for agent
        cache_key = f"{agent_type}_{env_inst.name}_{agent_source}"

        def get_current_agent():
            if agent_source == "Agent non entraîné (aléatoire)":
                return create_ui_agent(agent_type, env_inst, seed=42)
            if cache_key in st.session_state.trained_agents_cache:
                return st.session_state.trained_agents_cache[cache_key]
            # Entraînement rapide automatique si non en cache
            with st.spinner(f"Préparation de l'agent {agent_type}..."):
                ag = create_ui_agent(agent_type, env_inst, seed=42)
                train_ui_agent(ag, env_inst, n_episodes=150, n_demos=30)
                st.session_state.trained_agents_cache[cache_key] = ag
                return ag

        btn_reinit = st.button("🔄 Réinitialiser l'environnement", use_container_width=True)
        if btn_reinit or (st.session_state.env.name != env_inst.name):
            st.session_state.env = env_inst
            st.session_state.env.reset()
            st.session_state.agent = get_current_agent()
            st.session_state.trajectory = [st.session_state.env.agent_pos]
            st.session_state.total_reward = 0.0
            st.session_state.step_count = 0
            st.session_state.done = False
            st.session_state.last_action = "Aucune (réinitialisé)"
            st.session_state.status_message = "Environnement réinitialisé."

        if st.session_state.agent is None:
            st.session_state.agent = get_current_agent()

        st.markdown("---")
        st.subheader("🎮 Commandes")
        col_ctrl1, col_ctrl2 = st.columns(2)
        with col_ctrl1:
            btn_step = st.button("➡️ Étape suivante", use_container_width=True, disabled=st.session_state.done)
        with col_ctrl2:
            btn_run_all = st.button("🚀 Exécuter épisode", use_container_width=True, disabled=st.session_state.done)

        sim_delay = st.slider("Délai animation (s)", 0.05, 0.5, 0.15, step=0.05)

    with col_view:
        st.subheader("🗺️ Grille d'état")

        # Action: Étape suivante
        if btn_step and not st.session_state.done:
            new_pos, reward, done, info, action_name = step_simulation(
                st.session_state.env, st.session_state.agent
            )
            st.session_state.trajectory.append(new_pos)
            st.session_state.total_reward += reward
            st.session_state.step_count += 1
            st.session_state.done = done
            st.session_state.last_action = action_name

            event = info.get("event", "step")
            if event == "goal":
                st.session_state.status_message = "🎯 Objectif atteint avec succès !"
            elif event == "obstacle":
                st.session_state.status_message = "💥 Collision avec un obstacle (pénalité -2.0)"
            elif event == "wall":
                st.session_state.status_message = "🧱 Heurt du mur bordier (pénalité -0.5, maintien)"
            elif event == "timeout":
                st.session_state.status_message = "⏱️ Limite maximale d'étapes atteinte."
            else:
                st.session_state.status_message = f"Déplacement vers {new_pos}."

        # Action: Exécution complète de l'épisode avec animation en direct
        if btn_run_all and not st.session_state.done:
            placeholder_fig = st.empty()
            placeholder_msg = st.empty()
            while not st.session_state.done:
                new_pos, reward, done, info, action_name = step_simulation(
                    st.session_state.env, st.session_state.agent
                )
                st.session_state.trajectory.append(new_pos)
                st.session_state.total_reward += reward
                st.session_state.step_count += 1
                st.session_state.done = done
                st.session_state.last_action = action_name

                # Rendu de l'étape courante
                fig_step = render_grid_figure(
                    st.session_state.env,
                    current_pos=new_pos,
                    trajectory=st.session_state.trajectory,
                    title=f"{st.session_state.agent.name} — Étape {st.session_state.step_count}",
                )
                placeholder_fig.pyplot(fig_step, use_container_width=False)
                plt.close(fig_step)

                event = info.get("event", "step")
                if event == "goal":
                    st.session_state.status_message = "🎯 Objectif atteint avec succès !"
                elif event == "timeout":
                    st.session_state.status_message = "⏱️ Limite d'étapes atteinte."
                placeholder_msg.info(st.session_state.status_message)
                time.sleep(sim_delay)

        # Affichage standard de la grille
        fig = render_grid_figure(
            st.session_state.env,
            current_pos=st.session_state.env.agent_pos,
            trajectory=st.session_state.trajectory,
            title=f"Environnement : {st.session_state.env.name} | Agent : {st.session_state.agent.name}",
        )
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)

        # Panneau de télémétrie
        st.markdown("### 📊 Télémétrie de l'Agent")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Étapes effectuées", f"{st.session_state.step_count} / {st.session_state.env.max_steps}")
        m2.metric("Récompense cumulée", f"{st.session_state.total_reward:.2f}")
        m3.metric("Dernière action", st.session_state.last_action)
        m4.metric("Statut", "Terminé" if st.session_state.done else "En cours")

        if st.session_state.done:
            if st.session_state.env.agent_pos == st.session_state.env.goal_pos:
                st.success(st.session_state.status_message)
            else:
                st.warning(st.session_state.status_message)
        else:
            st.info(st.session_state.status_message)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE B : Entraînement dynamique
# ─────────────────────────────────────────────────────────────────────────────
elif page == "⚡ Page B — Entraînement":
    st.header("⚡ Page B — Entraînement en Direct & Suivi Métrique")
    st.markdown(
        "Configurez et lancez l'entraînement d'un agent. Visualisez en direct la convergence "
        "du taux de réussite et de la récompense moyenne au fil des épisodes."
    )

    col_params, col_metrics = st.columns([1, 2])

    with col_params:
        st.subheader("⚙️ Paramètres d'apprentissage")
        train_env_name = st.selectbox(
            "Environnement",
            list(PRESET_ENVS.keys()),
            index=0,
            key="train_env_choice",
        )
        train_agent_type = st.selectbox(
            "Agent à entraîner",
            ["Q-Learning", "Behavioral Cloning", "Hybrid ORL"],
            index=2,
            key="train_agent_choice",
        )
        n_episodes = st.slider("Nombre d'épisodes", 50, 1000, 200, step=25, key="train_nepisodes")
        seed = st.number_input("Graine aléatoire (seed)", value=42, step=1, key="train_seed")

        st.markdown("#### Hyperparamètres avancés")
        lr = st.number_input("Taux d'apprentissage (RL lr)", value=0.1, min_value=0.001, max_value=1.0, step=0.01)
        discount = st.number_input("Facteur d'actualisation (gamma)", value=0.99, min_value=0.8, max_value=0.999, step=0.01)

        if train_agent_type in ["Behavioral Cloning", "Hybrid ORL"]:
            st.markdown("#### Démonstrations d'apprentissage social")
            n_demos = st.slider("Nombre de démonstrations", 5, 100, 30, step=5)
            demo_quality = st.selectbox("Qualité des démonstrations", ["optimal", "suboptimal"], index=0)
            noise_prob = st.slider("Taux de bruit (si suboptimal)", 0.0, 0.5, 0.2, step=0.05) if demo_quality == "suboptimal" else 0.0
        else:
            n_demos = 0
            demo_quality = "optimal"
            noise_prob = 0.0

        btn_start_train = st.button("🚀 Lancer l'entraînement", use_container_width=True, type="primary")

    with col_metrics:
        st.subheader("📈 Suivi en Temps Réel")

        status_box = st.empty()
        progress_bar = st.empty()
        chart_sr = st.empty()
        chart_r = st.empty()
        final_summary = st.empty()

        if not btn_start_train:
            status_box.info("ℹ️ Configurez les paramètres et cliquez sur **Lancer l'entraînement**.")
        else:
            status_box.warning("⏳ **Statut : Entraînement en cours...**")
            progress_bar.progress(0)

            # Instanciation de l'environnement et de l'agent
            env_train = create_ui_environment(preset_name=train_env_name)
            agent_train = create_ui_agent(
                agent_type=train_agent_type,
                env=env_train,
                seed=int(seed),
                learning_rate=float(lr),
                discount_factor=float(discount),
            )

            # Historiques pour mise à jour graphique
            history_episodes = []
            history_sr = []
            history_reward = []

            def on_progress(current_ep, total_ep, sr, r):
                progress = current_ep / total_ep
                progress_bar.progress(progress)
                history_episodes.append(current_ep)
                history_sr.append(sr)
                history_reward.append(r)

                df_curves = pd.DataFrame({
                    "Épisode": history_episodes,
                    "Taux de Succès (glissant)": history_sr,
                    "Récompense Moyenne": history_reward,
                }).set_index("Épisode")

                chart_sr.line_chart(df_curves["Taux de Succès (glissant)"], height=200)
                chart_r.line_chart(df_curves["Récompense Moyenne"], height=200)

            t0 = time.time()
            history = train_ui_agent(
                agent=agent_train,
                env=env_train,
                n_episodes=int(n_episodes),
                n_demos=n_demos,
                demo_quality=demo_quality,
                noise_prob=noise_prob,
                progress_callback=on_progress,
            )
            duration = time.time() - t0

            progress_bar.progress(1.0)
            status_box.success(f"✅ **Statut : Entraînement terminé avec succès en {duration:.2f} secondes !**")

            # Sauvegarder dans le cache pour Page A
            st.session_state.trained_agents_cache[f"{train_agent_type}_{env_train.name}_custom"] = agent_train

            # Évaluation finale sans exploration
            eval_metrics = evaluate_agent(agent_train, env_train, n_episodes=50)
            conv_ep = history.episodes_to_threshold(threshold=0.7, window=30)

            with final_summary.container():
                st.markdown("### 🏆 Bilan de Performance")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Taux de réussite final", f"{eval_metrics['success_rate']:.1%}")
                c2.metric("Récompense moyenne", f"{eval_metrics['mean_reward']:.2f} ± {eval_metrics['std_reward']:.2f}")
                c3.metric("Nombre moyen d'étapes", f"{eval_metrics['mean_steps']:.1f}")
                c4.metric("Convergence (70%)", f"Épisode {conv_ep}" if conv_ep is not None else "Non atteinte")

                st.info(
                    "💡 Cet agent a été mis en cache. Vous pouvez le visualiser immédiatement "
                    "dans la **Page A — Visualisation GridWorld** !"
                )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE C : Comparaison simultanée des trois paradigmes
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📊 Page C — Comparaison":
    st.header("📊 Page C — Comparaison Rigoureuse des Trois Stratégies")
    st.markdown(
        "Ce banc expérimental compare simultanément **Pure RL**, **Pure Imitation (BC)** et **Hybrid ORL** "
        "sur des environnements rigoureusement identiques avec des conditions d'échantillonnage équitables."
    )

    c_conf1, c_conf2, c_conf3 = st.columns(3)
    with c_conf1:
        comp_env_name = st.selectbox("Environnement de test", list(PRESET_ENVS.keys()), index=0)
    with c_conf2:
        comp_n_episodes = st.slider("Nombre d'épisodes d'interaction", 50, 600, 150, step=25)
    with c_conf3:
        comp_seed = st.number_input("Graine de reproductibilité", value=42, step=1)

    btn_compare = st.button("🔬 Lancer le Benchmark Comparatif", type="primary", use_container_width=True)

    if btn_compare:
        env_bench = create_ui_environment(preset_name=comp_env_name)

        agents_to_test = {
            "Pure RL (Q-Learning)": create_ui_agent("Q-Learning", env_bench, seed=int(comp_seed)),
            "Pure Imitation (BC)": create_ui_agent("Behavioral Cloning", env_bench, seed=int(comp_seed)),
            "Hybrid ORL (BC->QL)": create_ui_agent("Hybrid ORL", env_bench, seed=int(comp_seed)),
        }

        results_summary = []
        histories_dict = {}

        progress_comp = st.progress(0)
        status_comp = st.empty()

        for idx, (agent_label, ag) in enumerate(agents_to_test.items()):
            status_comp.info(f"⏳ Entraînement de l'agent **{agent_label}**...")
            h = train_ui_agent(
                agent=ag,
                env=env_bench,
                n_episodes=int(comp_n_episodes),
                n_demos=30,
                demo_quality="optimal",
            )
            histories_dict[agent_label] = h
            eval_res = evaluate_agent(ag, env_bench, n_episodes=50)
            conv = h.episodes_to_threshold(threshold=0.7, window=30)

            results_summary.append({
                "Stratégie": agent_label,
                "Taux de Succès Test": f"{eval_res['success_rate']:.1%}",
                "Récompense Moyenne": f"{eval_res['mean_reward']:.2f} ± {eval_res['std_reward']:.2f}",
                "Étapes Moyennes": f"{eval_res['mean_steps']:.1f}",
                "Convergence (70%)": f"Épisode {conv}" if conv is not None else "Non atteinte",
                "_raw_sr": eval_res["success_rate"],
                "_raw_r": eval_res["mean_reward"],
                "_raw_steps": eval_res["mean_steps"],
                "_conv_ep": conv if conv is not None else comp_n_episodes,
            })
            progress_comp.progress((idx + 1) / len(agents_to_test))

        status_comp.success("✅ **Benchmark comparatif achevé avec succès !**")

        st.subheader("📋 Tableau Récapitulatif")
        df_summary = pd.DataFrame(results_summary)
        st.dataframe(
            df_summary[["Stratégie", "Taux de Succès Test", "Récompense Moyenne", "Étapes Moyennes", "Convergence (70%)"]],
            use_container_width=True,
        )

        st.subheader("📈 Courbes d'Apprentissage Comparatives")
        fig_comp, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=100)

        colors = {"Pure RL (Q-Learning)": "#E74C3C", "Pure Imitation (BC)": "#3498DB", "Hybrid ORL (BC->QL)": "#2ECC71"}
        styles = {"Pure RL (Q-Learning)": "-", "Pure Imitation (BC)": "--", "Hybrid ORL (BC->QL)": "-."}

        for agent_label, h in histories_dict.items():
            records = h.records
            window = 20
            rolling_sr = []
            rolling_r = []
            for i in range(len(records)):
                sub = records[max(0, i - window + 1): i + 1]
                rolling_sr.append(sum(r.success for r in sub) / len(sub))
                rolling_r.append(float(np.mean([r.total_reward for r in sub])))

            ep_range = range(len(records))
            ax1.plot(ep_range, rolling_sr, label=agent_label, color=colors[agent_label], linestyle=styles[agent_label], lw=2)
            ax2.plot(ep_range, rolling_r, label=agent_label, color=colors[agent_label], linestyle=styles[agent_label], lw=2)

        ax1.set_title("Taux de Succès Glissant", fontsize=11, fontweight="bold")
        ax1.set_xlabel("Épisode")
        ax1.set_ylabel("Taux de Succès")
        ax1.set_ylim(-0.05, 1.05)
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend(fontsize=9)

        ax2.set_title("Récompense Moyenne Glissante", fontsize=11, fontweight="bold")
        ax2.set_xlabel("Épisode")
        ax2.set_ylabel("Récompense")
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend(fontsize=9)

        plt.tight_layout()
        st.pyplot(fig_comp, use_container_width=True)
        plt.close(fig_comp)

        st.markdown("### 🔍 Analyse Scientifique")
        st.markdown(
            "- **Pure RL (Q-Learning)** : Démarre avec un taux de succès nul en exploration $\\epsilon=1.0$ pure, "
            "nécessitant de nombreux épisodes pour localiser l'objectif dans les environnements creux.\n"
            "- **Pure Imitation (BC)** : Taux de succès initial maximal dès l'épisode 0, mais reste vulnérable "
            "au *covariate shift* (Ross & Bagnell, 2010) si l'agent est perturbé hors des états de démonstrations.\n"
            "- **Hybrid ORL** : Bénéficie immédiatement du warm-start observationnel tout en exploitant "
            "les mises à jour de Bellman pour garantir la robustesse asymptotique."
        )
    else:
        st.info("ℹ️ Cliquez sur **Lancer le Benchmark Comparatif** pour exécuter les trois modèles sur l'environnement choisi.")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE D : Résultats Réels & Figures
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📁 Page D — Résultats & Figures":
    st.header("📁 Page D — Résultats Enregistrés & Galerie de Figures")
    st.markdown(
        "Explorez les données réelles enregistrées dans `results/` et les visualisations scientifiques "
        "générées dans `figures/`. Aucun résultat simulé n'est affiché."
    )

    tab_csv, tab_figures, tab_runner = st.tabs(["📊 Données CSV réelles", "🖼️ Galerie de Figures", "⚙️ Ré-exécuter le Benchmark"])

    with tab_csv:
        st.subheader("Fichiers de Résultats (`results/`)")
        csv_data = load_csv_results(RESULTS_DIR)

        if not csv_data:
            st.warning("Aucun fichier CSV trouvé dans `results/`. Lancez une expérimentation pour générer des données.")
        else:
            selected_csv = st.selectbox("Sélectionnez un fichier de résultats :", list(csv_data.keys()))
            df_cur = csv_data[selected_csv]

            st.write(f"**Lignes :** {len(df_cur)} | **Colonnes :** {len(df_cur.columns)}")
            st.dataframe(df_cur, use_container_width=True)

            csv_bytes = df_cur.to_csv(index=False, encoding="utf-8").encode("utf-8")
            st.download_button(
                label=f"📥 Télécharger {selected_csv}",
                data=csv_bytes,
                file_name=selected_csv,
                mime="text/csv",
                use_container_width=True,
            )

    with tab_figures:
        st.subheader("Figures Scientifiques Publiables (`figures/`)")
        figs_list = list_available_figures(FIGURES_DIR)

        if not figs_list:
            st.warning("Aucune figure trouvée dans `figures/`. Utilisez l'onglet 'Ré-exécuter le Benchmark' pour les générer.")
        else:
            cols = st.columns(2)
            for idx, fig_path in enumerate(figs_list):
                with cols[idx % 2]:
                    st.image(str(fig_path), caption=fig_path.name, use_container_width=True)
                    with open(fig_path, "rb") as f:
                        img_bytes = f.read()
                    st.download_button(
                        label=f"📥 Télécharger {fig_path.name}",
                        data=img_bytes,
                        file_name=fig_path.name,
                        mime="image/png",
                        key=f"dl_{fig_path.name}",
                    )

    with tab_runner:
        st.subheader("Générer l'ensemble des résultats et figures")
        st.markdown(
            "Cette commande exécute le script `scripts/run_experiments.py` officiel du projet, "
            "exporte tous les fichiers CSV dans `results/` et produit toutes les figures dans `figures/`."
        )

        col_run1, col_run2 = st.columns([1, 1])
        with col_run1:
            run_mode = st.radio("Mode d'exécution :", ["quick (validation rapide, 1 graine, 200 épisodes)", "full (benchmark complet, 3 graines, tous les envs)"])
        with col_run2:
            st.write("")
            st.write("")
            btn_exec_script = st.button("⚡ Lancer `run_experiments.py`", type="primary", use_container_width=True)

        if btn_exec_script:
            mode_arg = "quick" if "quick" in run_mode else "full"
            with st.spinner(f"Exécution de `python scripts/run_experiments.py --mode {mode_arg}` en cours..."):
                import subprocess
                cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "run_experiments.py"), "--mode", mode_arg]
                res = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, encoding="utf-8")

                if res.returncode == 0:
                    st.success("✅ Expériences terminées avec succès !")
                    with st.expander("Consulter la sortie standard (stdout)"):
                        st.code(res.stdout)
                    st.rerun()
                else:
                    st.error(f"Erreur lors de l'exécution (code {res.returncode}) :")
                    st.code(res.stderr or res.stdout)
