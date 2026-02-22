import pandas as pd
import numpy as np
from datetime import time, datetime

# =============================================================================
# GOBERNANZA DE ACTIVOS Y ESPECIFICACIONES
# =============================================================================
ASSET_SPECS = {
    "NQ": {"tick_size": 0.25, "tick_value": 5.0, "points_full_value": 20.0, "commission": 10.40, "avg_slippage_ticks": 2},
    "ES": {"tick_size": 0.25, "tick_value": 12.5, "points_full_value": 50.0, "commission": 10.40, "avg_slippage_ticks": 1},
    "YM": {"tick_size": 1.0, "tick_value": 5.0, "points_full_value": 5.0, "commission": 10.40, "avg_slippage_ticks": 1},
    "GC": {"tick_size": 0.1, "tick_value": 10.0, "points_full_value": 100.0, "commission": 13.40, "avg_slippage_ticks": 1},
    "CL": {"tick_size": 0.01, "tick_value": 10.0, "points_full_value": 1000.0, "commission": 13.40, "avg_slippage_ticks": 2}
}

class StrategyConfig:
    """Configuración inmutable de la operativa y gestión de riesgo."""
    def __init__(self, asset_name="NQ", risk_usd=2000, reward_ratio=2.0, be_trigger_r=1.0, 
                 be_offset_ticks=0,max_trades_per_day=-1,trading_windows=[("09:30", "10:30")],
                 force_close_time="15:56", direction="Both",execution_mode="Optimista"):
        """
        max_trades_per_day: -1 para infinito, o un entero positivo para limitar trades diarios.
        """
        spec = ASSET_SPECS.get(asset_name, ASSET_SPECS["NQ"])
        self.asset_name = asset_name
        self.risk_usd = risk_usd
        self.reward_ratio = reward_ratio
        self.be_trigger_r = be_trigger_r
        self.tick_size = spec["tick_size"]
        self.tick_value = spec["tick_value"]
        self.points_full_value = spec["points_full_value"]
        self.comm_per_side = spec["commission"] / 2
        self.slippage_points = spec["avg_slippage_ticks"] * self.tick_size
        self.be_offset = be_offset_ticks * self.tick_size
        self.max_trades_per_day = max_trades_per_day
        self.trading_windows = trading_windows
        self.force_close_time = force_close_time
        self.direction = direction
        self.execution_mode = execution_mode

class AuditLogger:
    """Utilidad para impresión de logs detallados y depuración."""
    @staticmethod
    def log_trade_start(t):
        print(f"\n🚀 [TRADE #{t['id']}] {t['type']} ENTERED @ {t['entry']:.2f}")
        print(f"   Initial SL: {t['sl']:.2f} | Initial TP: {t['tp']:.2f} | Qty: {t['qty']}")

    @staticmethod
    def log_candle(dt, row, t):
        print(f"   🕒 {dt.time()} | O:{row['Open_adj']:.2f} H:{row['High_adj']:.2f} L:{row['Low_adj']:.2f} C:{row['Close_adj']:.2f} | SL:{t['sl']:.2f} BE:{t['be_active']}")

    @staticmethod
    def log_trade_end(t, exit_price, pnl_r, reason):
        icon = "✅" if pnl_r > 0 else "❌" if pnl_r < 0 else "⚪"
        print(f"🏁 {icon} [TRADE #{t['id']}] CLOSED @ {exit_price:.2f} | PnL: {pnl_r:.2f}R | Reason: {reason}")

class QuantEngineV2:
    def __init__(self, df, config):
        self.df = df
        self.config = config
        
        # 1. Buscamos las especificaciones del activo (GC en este caso)
        # Si config no tiene asset_name, usamos "GC" por defecto
        asset_name = getattr(config, 'asset_name', 'GC')
        spec = ASSET_SPECS.get(asset_name, ASSET_SPECS["GC"])
        
        # 2. Registramos los atributos que el método _round_to_tick va a buscar
        self.tick_size = spec["tick_size"]
        self.tick_value = spec["tick_value"]
        
        # 3. Inicializadores de estado
        self.trades = []
        self.current_day_trades = 0
        self.last_date = None


        
    def _is_in_trading_window(self, current_dt):
        if not self.config.trading_windows:
            return True
        
        # Extraemos la hora (asegurándonos de ignorar microsegundos)
        t_obj = current_dt.time() if hasattr(current_dt, 'time') else current_dt
        curr_time = time(t_obj.hour, t_obj.minute, 0) # Normalizamos a HH:MM:00
        
        for start_str, end_str in self.config.trading_windows:
            # Convertimos los límites de la config
            t_start = time.fromisoformat(start_str)
            t_end = time.fromisoformat(end_str)
            
            # Lógica TV: Desde el segundo 0 de la apertura hasta el segundo 0 del cierre
            if t_start <= curr_time <= t_end:
                return True
        return False
        
    def _round_to_tick(self, price):
        """Este es el método que fallaba por falta de self.tick_size"""
        if pd.isna(price): 
            return price
        return np.round(price / self.tick_size) * self.tick_size

    def _resolve_intra_candle(self, row, t):
        """
        Resuelve la salida del trade según el modo de ejecución configurado.
        t: diccionario del trade activo.
        """
        high, low, open_p, close_p = row['High_adj'], row['Low_adj'], row['Open_adj'], row['Close_adj']
        ptype, sl, tp = t['type'], t['sl'], t['tp']
        mode = self.config.execution_mode  # 'Optimista', 'Pesimista', 'OHLC'

        # --- MODO OPTIMISTA: Prioriza el Take Profit ---
        if mode == "Optimista":
            if ptype == 1: # Long
                if high >= tp: return "TP"
                if low <= sl: return "SL"
            else: # Short
                if low <= tp: return "TP"
                if high >= sl: return "SL"

        # --- MODO PESIMISTA: Prioriza el Stop Loss ---
        elif mode == "Pesimista":
            if ptype == 1: # Long
                if low <= sl: return "SL"
                if high >= tp: return "TP"
            else: # Short
                if high >= sl: return "SL"
                if low <= tp: return "TP"

        # --- MODO OHLC: Sigue la secuencia lógica de la vela ---
        elif mode == "OHLC":
            # Determinamos si la vela es alcista o bajista para estimar el recorrido interno
            if ptype == 1: # Long
                if open_p >= tp: return "TP" # Gana en la apertura
                if open_p <= sl: return "SL" # Pierde en la apertura
                
                if close_p >= open_p: # Vela Verde: Open -> Low -> High -> Close
                    if low <= sl: return "SL"
                    if high >= tp: return "TP"
                else: # Vela Roja: Open -> High -> Low -> Close
                    if high >= tp: return "TP"
                    if low <= sl: return "SL"
            else: # Short
                if open_p <= tp: return "TP"
                if open_p >= sl: return "SL"
                
                if close_p <= open_p: # Vela Roja (Bajista): Open -> High -> Low -> Close
                    if low <= tp: return "TP"
                    if high >= sl: return "SL"
                else: # Vela Verde (Alcista): Open -> Low -> High -> Close
                    if high >= sl: return "SL"
                    if low <= tp: return "TP"

        return None

    def run(self, start_date=None, end_date=None, verbose=False):
        # 1. RESET DE ESTADO
        self.trades = []
        self.current_day_trades = 0
        self.last_date = None
        
        df_proc = self.df.copy()
        if start_date: df_proc = df_proc[df_proc['Timestamp_NY'] >= start_date]
        if end_date: df_proc = df_proc[df_proc['Timestamp_NY'] <= end_date]
        df_proc = df_proc.reset_index(drop=True)

        in_pos = False
        t = {}

        for i, row in df_proc.iterrows():
            curr_dt = row['Timestamp_NY']
            curr_date = curr_dt.date()
            curr_time = curr_dt.time()
            
            # Mantenemos el contador vivo para el reporte, pero no para validar
            if self.last_date is None or curr_date != self.last_date:
                self.current_day_trades = 0
                self.last_date = curr_date

            # --- SECCIÓN A: SALIDAS ---
            if in_pos:
                if verbose: AuditLogger.log_candle(curr_dt, row, t)
                res = self._resolve_intra_candle(row, t)
                
                if not t['be_active']:
                    dist = (row['High_adj'] - t['entry']) if t['type'] == 1 else (t['entry'] - row['Low_adj'])
                    if dist >= t['risk_pts'] * self.config.be_trigger_r:
                        t['be_active'] = True
                        t['sl'] = t['entry'] + (self.config.be_offset * t['type'])

                force_exit = (curr_time >= time.fromisoformat(self.config.force_close_time))
                
                if res or force_exit:
                    exit_raw = t['sl'] if res == "SL" else (t['tp'] if res == "TP" else row['Close_adj'])
                    slippage = self.config.slippage_points if res != "TP" else 0
                    exit_final = exit_raw - (slippage * t['type'])

                    
                    # 1. Cálculos de PnL
                    # 1. Calculamos cuánto dinero se arriesgó realmente en este trade (R inicial)
                    # Riesgo en puntos * Cantidad * Valor del punto
                    usd_risk_at_stake = (t['risk_pts'] * self.config.points_full_value * t['qty'])
                    
                    # 2. PnL neto en USD (Esto ya lo tienes bien, incluye comisiones)
                    pnl_usd = ((exit_final - t['entry']) * t['type'] * t['qty'] * self.config.points_full_value) - (t['qty'] * self.config.comm_per_side * 2)
                    
                    # 3. PnL en unidades de R (La corrección)
                    # Ahora dividimos por el riesgo del trade, no por el capital de la cuenta
                    pnl_r_value = pnl_usd / usd_risk_at_stake

                    # 2. Registro en la lista de trades
                    self.trades.append({
                        'id': len(self.trades) + 1, 
                        'date': self.last_date, 
                        'entry_time': t['time'], 
                        'exit_time': curr_dt,
                        'type': "Long" if t['type'] == 1 else "Short", 
                        'qty': t['qty'],
                        'entry': t['entry'], 
                        'exit': exit_final, 
                        'pnl_usd': pnl_usd,
                        'pnl_r': pnl_r_value, 
                        'reason': res or "ForceClose"
                    })

                    # 3. Reporte Detallado (Aquí es donde estaba el error)
                    if verbose:
                        emoji = '✅' if res == 'TP' else '❌'
                        print(f"🏁 {emoji} [TRADE #{len(self.trades)}] "
                              f"CLOSED @ {exit_final:.2f} | PnL: {pnl_r_value:.2f}R | Reason: {res or 'ForceClose'}")
                        # Mostramos la hora de entrada y salida para contraste
                        print(f"   ⏱️ ENTRADA: {t['time'].strftime('%H:%M:%S')} | SALIDA: {curr_dt.strftime('%H:%M:%S')}")
                        print(f"   💰 PnL USD: ${pnl_usd:.2f} ") # Para comparar con los $1,729.6 de la imagen
                    
                    in_pos = False
                
                if in_pos: continue

# --- SECCIÓN B: ENTRADAS (Bypass de maxtrades) ---
            # 1. Inicialización de seguridad (Evita UnboundLocalError)
            is_long = False
            is_short = False

            
            # 1. Validación ultra-precisa de ventana
            in_window = self._is_in_trading_window(curr_dt)
            
            # 2. LOG DE AUDITORÍA (Solo para el rango de apertura 09:30 - 09:35)
            if curr_dt.hour == 9 and 30 <= curr_dt.minute <= 35:
                sig_l = row.get('sig_long', False)
                sig_s = row.get('sig_short', False)
                if verbose:
                    print(f"🔍 [AUDIT {curr_dt.time()}] Window: {in_window} | InPos: {in_pos} | SigL: {sig_l} | SigS: {sig_s}")
            
            # 2. Solo evaluamos si NO estamos en posición y el guardia da permiso
            if not in_pos and in_window:
                is_long = row.get('sig_long', False) and self.config.direction in ["Long", "Both"]
                is_short = row.get('sig_short', False) and self.config.direction in ["Short", "Both"]

            # 3. Lógica de ejecución (Motor Desacoplado)
            if is_long or is_short:
                ptype = 1 if is_long else -1
                
                # 1. ENTRADA: Se calcula al momento (Open de la siguiente vela o Close + slippage)
                # Aplicamos redondeo al tick para evitar basura de punto flotante
                entry_p = self._round_to_tick(row['Close_adj'] + (self.config.slippage_points * ptype))
                
                # 2. NIVELES ESTRATÉGICOS: Tomados directamente del DataFrame
                # Estos ya deben venir calculados según la lógica de Pine (basados en la señal)
                sl_p = self._round_to_tick(row['sl_level'])
                tp_p = self._round_to_tick(row['tp_level'])
                
                # 3. RIESGO DE EJECUCIÓN: Distancia real entre nuestra entrada y el SL
                # Se usa exclusivamente para el cálculo del QTY (Gestión de Riesgo)
                risk_execution_pts = abs(entry_p - sl_p)
                
                if risk_execution_pts > 0:
                    # Cálculo de valor de riesgo por contrato usando los specs del activo
                    risk_usd_contract = (risk_execution_pts / self.tick_size) * self.tick_value
                    qty = int(np.floor(self.config.risk_usd / risk_usd_contract))
                    
                    if qty >= 1:
                        in_pos = True
                        self.current_day_trades += 1 
                        
                        t = {
                            'id': len(self.trades) + 1, 
                            'type': ptype, 
                            'entry': entry_p, 
                            'sl': sl_p, 
                            'tp': tp_p,   # Nivel exacto heredado de la estrategia
                            'qty': qty, 
                            'risk_pts': risk_execution_pts, 
                            'be_active': False, 
                            'time': curr_dt
                        }
                        if verbose: AuditLogger.log_trade_start(t)

                    elif verbose:
                        print(f"    ⚠️ QTY=0 en {curr_dt}: Riesgo ${risk_usd_contract:.2f} excede presupuesto.")
        
        return pd.DataFrame(self.trades)


    
def analyze_specific_day(engine, target_date_str):
    """Ejecuta el backtest solo para un día y muestra el log detallado."""
    print(f"\n--- AUDITORÍA DETALLADA: {target_date_str} ---")
    start = datetime.strptime(target_date_str + " 00:00:00", "%Y-%m-%d %H:%M:%S")
    end = datetime.strptime(target_date_str + " 23:59:59", "%Y-%m-%d %H:%M:%S")
    return engine.run(start_date=start, end_date=end, verbose=True)