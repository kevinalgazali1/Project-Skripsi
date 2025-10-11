<?php

namespace App\Http\Middleware;

use Closure;
use Illuminate\Support\Facades\Auth;

class RedirectIfAuthenticatedCustom
{
    public function handle($request, Closure $next, ...$guards)
    {
        $user = Auth::user();

        if ($user) {
            // Redirect berdasarkan role
            switch ($user->role) {
                case 'mitra':
                    return redirect()->route('mitra.index');
                case 'alumnisiswa':
                    return redirect()->route('alumni-siswa.index');
                case 'admin':
                    return redirect('/admin/dashboard'); // Sesuaikan jika ada route admin
            }
        }

        return $next($request);
    }
}
