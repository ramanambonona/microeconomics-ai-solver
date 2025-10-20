// Configuration pour les environnements de développement et production
const config = {
    development: {
        apiUrl: 'http://localhost:8000'
    },
    production: {
        apiUrl: 'https://sde-backend-url.railway.app' // Remplacez par votre URL backend
    }
};

const environment = window.location.hostname === 'localhost' ? 'development' : 'production';
export const API_URL = config[environment].apiUrl;