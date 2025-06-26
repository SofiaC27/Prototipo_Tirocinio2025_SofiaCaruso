import subprocess
import signal
import shutil


class MCPServer:
    def __init__(self):
        self.process = None

    def start(self):
        """
        Metodo per avviare il server MCP tramite il comando 'npx @sqlitecloud/mcp-server'
        - Cerca automaticamente il percorso eseguibile del comando 'npx' nel sistema operativo
        - Se 'npx' non è trovato nel PATH, solleva un FileNotFoundError
        - Avvia il server MCP come processo figlio del processo Python
        - Utilizza 'shell=True' per compatibilità con ambienti Windows (.cmd)
        - Redirige lo stdout e stderr per eventuali log o debugging
        - Salva il processo avviato nell'attributo 'self.process' per poterlo chiudere in seguito
        - Stampa il PID del server MCP avviato per monitoraggio
        """
        npx_path = shutil.which("npx")

        if not npx_path:
            raise FileNotFoundError("npx not found in the system PATH")

        self.process = subprocess.Popen(
            [npx_path, "@sqlitecloud/mcp-server"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True  # ESSENZIALE su windows per .cmd
        )
        print(f"[MCP Server] Started (PID: {self.process.pid})")

    def stop(self):
        """
        Metodo per terminare correttamente il server MCP precedentemente avviato
        - Verifica se esiste un processo attivo salvato in 'self.process'
        - Invia un segnale di terminazione al processo MCP (terminate)
        - Attende che il processo sia completamente chiuso (wait)
        - Stampa conferma dell’arresto del server MCP
        - Non fa nulla se 'self.process' è None (server non avviato)
        """
        if self.process:
            print(f"[MCP Server] Terminating MCP...")
            self.process.terminate()
            self.process.wait()
            print("[MCP Server] Stopped.")
