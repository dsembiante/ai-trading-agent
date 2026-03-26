from database import Database

db = Database()
trades = db.get_all_trades()

print(f'Total trades in database: {len(trades)}')

closed = [t for t in trades if t.get('status') == 'closed']
open_t = [t for t in trades if t.get('status') == 'open']

print(f'Open trades: {len(open_t)}')
print(f'Closed trades: {len(closed)}')

if closed:
    print('\nFirst 3 closed trades:')
    for t in closed[:3]:
        print(f"  {t.get('ticker')} | {t.get('trade_type')} | P&L: {t.get('pnl')} | Exit: {t.get('exit_price')}")
else:
    print('\nNo closed trades found in database')

if open_t:
    print('\nOpen trades:')
    for t in open_t:
        print(f"  {t.get('ticker')} | {t.get('hold_period')} | Entry: {t.get('entry_price')}")
