import subprocess
import shutil
import os


class MCPServer:
    def __init__(self, db_path="documents.db", port=8080):
        self.process = None
        self.db_path = os.path.abspath(db_path)  # percorso assoluto del file DB
        self.port = port

    def start(self):
        """
        Metodo per avviare il server MCP tramite il comando 'npx @sqlitecloud/mcp-server'
        - Cerca automaticamente il percorso eseguibile del comando 'npx' nel sistema operativo
        - Se 'npx' non è trovato nel PATH, solleva un FileNotFoundError
        - Costruisce il comando da eseguire specificando il file del database SQLite locale da esporre e la porta
          su cui il server MCP deve rimanere in ascolto
        - Avvia il server MCP come processo figlio del processo Python
        - Utilizza 'shell=True' per compatibilità con ambienti Windows (.cmd)
        - Redirige lo stdout e stderr per eventuali log o debugging
        - Salva il processo avviato nell'attributo 'self.process' per poterlo chiudere in seguito
        - Stampa il PID del server MCP avviato per monitoraggio
        """
        npx_path = shutil.which("npx")

        if not npx_path:
            raise FileNotFoundError("npx not found in the system PATH")

        command = [
            npx_path,
            "@sqlitecloud/mcp-server",
            "--db", self.db_path,
            "--port", str(self.port)
        ]

        self.process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True  # ESSENZIALE su windows per .cmd
        )

        print(f"[MCP Server] Started on port {self.port} with DB '{self.db_path}' (PID: {self.process.pid})")

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
