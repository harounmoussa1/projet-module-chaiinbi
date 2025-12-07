document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('configForm');
    const runBtn = document.getElementById('runBtn');
    const statusDiv = document.getElementById('status');
    const resultsTableBody = document.querySelector('#resultsTable tbody');
    const downloadBtn = document.getElementById('downloadBtn');
    let chartInstance = null;
    let currentParetoData = [];

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        // Disable button and show loading state
        runBtn.disabled = true;
        runBtn.textContent = 'Optimisation en cours...';
        downloadBtn.style.display = 'none'; // Hide download button during run
        statusDiv.textContent = 'Veuillez patienter, l\'algorithme génétique tourne...';
        statusDiv.style.color = '#94a3b8';

        const formData = new FormData(form);
        const data = {
            dataset: formData.get('dataset'),
            ngen: formData.get('ngen'),
            pop_size: formData.get('pop_size')
        };

        try {
            const response = await fetch('/api/run', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data)
            });

            const result = await response.json();

            if (result.status === 'success') {
                currentParetoData = result.pareto; // Store for CSV download
                updateChart(result.population, result.pareto);
                updateTable(result.pareto);
                statusDiv.textContent = 'Optimisation terminée avec succès !';
                statusDiv.style.color = '#10b981';
                downloadBtn.style.display = 'block'; // Show download button
            } else {
                statusDiv.textContent = 'Erreur: ' + result.message;
                statusDiv.style.color = '#ef4444';
            }
        } catch (error) {
            console.error('Error:', error);
            statusDiv.textContent = 'Erreur de communication avec le serveur.';
            statusDiv.style.color = '#ef4444';
        } finally {
            runBtn.disabled = false;
            runBtn.textContent = 'Lancer l\'Optimisation';
        }
    });

    downloadBtn.addEventListener('click', () => {
        if (!currentParetoData || currentParetoData.length === 0) return;

        // CSV Header
        const headers = ['Precision', 'Temps_Entrainement', 'N_Estimators', 'Max_Depth', 'Min_Samples_Split', 'Max_Features'];

        // CSV Rows
        const rows = currentParetoData.map(ind => [
            ind.accuracy.toFixed(6),
            ind.training_time.toFixed(6),
            ind.n_estimators,
            ind.max_depth,
            ind.min_samples_split,
            ind.max_features.toFixed(6)
        ]);

        // Combine header and rows
        const csvContent = [
            headers.join(','),
            ...rows.map(row => row.join(','))
        ].join('\n');

        // Create Blob and download link
        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.setAttribute('href', url);
        link.setAttribute('download', 'resultats_pareto.csv');
        link.style.visibility = 'hidden';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    });

    function updateChart(population, pareto) {
        const ctx = document.getElementById('paretoChart').getContext('2d');

        const popData = population.map(ind => ({
            x: ind.training_time,
            y: ind.accuracy
        }));

        const paretoData = pareto.map(ind => ({
            x: ind.training_time,
            y: ind.accuracy
        }));

        // Sort pareto data for line drawing
        paretoData.sort((a, b) => a.x - b.x);

        if (chartInstance) {
            chartInstance.destroy();
        }

        chartInstance = new Chart(ctx, {
            type: 'scatter',
            data: {
                datasets: [
                    {
                        label: 'Population Finale',
                        data: popData,
                        backgroundColor: 'rgba(148, 163, 184, 0.5)',
                        borderColor: 'rgba(148, 163, 184, 1)',
                        pointRadius: 3
                    },
                    {
                        label: 'Front de Pareto',
                        data: paretoData,
                        backgroundColor: 'rgba(239, 68, 68, 1)',
                        borderColor: 'rgba(239, 68, 68, 1)',
                        pointRadius: 5,
                        showLine: true,
                        fill: false,
                        tension: 0.1
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Temps d\'entraînement (s) [Minimiser]',
                            color: '#94a3b8'
                        },
                        grid: {
                            color: '#334155'
                        },
                        ticks: {
                            color: '#94a3b8'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Précision (Accuracy) [Maximiser]',
                            color: '#94a3b8'
                        },
                        grid: {
                            color: '#334155'
                        },
                        ticks: {
                            color: '#94a3b8'
                        }
                    }
                },
                plugins: {
                    legend: {
                        labels: {
                            color: '#f8fafc'
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: function (context) {
                                return `Acc: ${context.parsed.y.toFixed(4)}, Time: ${context.parsed.x.toFixed(4)}s`;
                            }
                        }
                    }
                }
            }
        });
    }

    function updateTable(pareto) {
        resultsTableBody.innerHTML = '';

        // Sort by accuracy descending
        pareto.sort((a, b) => b.accuracy - a.accuracy);

        pareto.forEach(ind => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${(ind.accuracy * 100).toFixed(2)}%</td>
                <td>${ind.training_time.toFixed(4)}</td>
                <td>${ind.n_estimators}</td>
                <td>${ind.max_depth}</td>
                <td>${ind.min_samples_split}</td>
                <td>${ind.max_features.toFixed(2)}</td>
            `;
            resultsTableBody.appendChild(row);
        });
    }
});
