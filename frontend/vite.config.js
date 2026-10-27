import { defineConfig } from 'vite';
export default defineConfig({server:{proxy:{'/health':'http://127.0.0.1:8000','/ready':'http://127.0.0.1:8000','/predict':'http://127.0.0.1:8000','/models':'http://127.0.0.1:8000'}},build:{chunkSizeWarningLimit:650}});
