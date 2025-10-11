<?php

namespace App\Http\Controllers\Auth;

use Illuminate\View\View;
use Illuminate\Http\Request;
use App\Http\Controllers\Controller;
use Illuminate\Support\Facades\Auth;
use Illuminate\Http\RedirectResponse;
use App\Http\Requests\Auth\LoginRequest;
use Illuminate\Validation\ValidationException;

class AuthenticatedSessionController extends Controller
{
    /**
     * Display the login view.
     */
    public function create(): View
    {
        return view('auth.login');
    }

    /**
     * Handle an incoming authentication request.
     */
    public function store(LoginRequest $request): RedirectResponse
    {
        try {
            $request->authenticate();
            $request->session()->regenerate();

            $user = $request->user();

            if ($user->role === "admin") {
                return redirect()->route('admin.dashboard');
            } elseif ($user->role === "mitra") {
                return $user->profile_complete
                    ? redirect()->route('mitra.index')
                    : redirect()->route('mitra.addProfile');
            } elseif ($user->role === "alumnisiswa") {
                return $user->profile_complete
                    ? redirect()->route('alumni-siswa.index')
                    : redirect()->route('alumni-siswa.addProfile');
            }


            return redirect()->back();
        } catch (ValidationException $e) {
            $loginType = $request->input('login_type');

            if ($loginType === 'admin') {
                return redirect()->route('login.admin')
                    ->withInput()
                    ->with('login_error', 'Nomor HP atau Password salah.');
            } elseif ($loginType === 'mitra') {
                return redirect()->route('login.mitra')
                    ->withInput()
                    ->with('login_error', 'Nomor HP atau Password salah.');
            } elseif ($loginType === 'alumnisiswa') {
                return redirect()->route('login.alumni')
                    ->withInput()
                    ->with('login_error', 'Nomor HP atau Password salah.');
            }

            // Fallback jika tidak diketahui
            return redirect('/')
                ->with('login_error', 'Login gagal.');
        }
    }

    /**
     * Destroy an authenticated session.
     */
    public function destroy(Request $request): RedirectResponse
    {
        Auth::guard('web')->logout();

        $request->session()->invalidate();

        $request->session()->regenerateToken();

        return redirect('/');
    }
}
